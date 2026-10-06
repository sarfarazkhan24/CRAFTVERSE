import torch
import json
import os
from .layers.calibration import TemperatureScaler
from .layers.mc_dropout import MCDropout
from .layers.ood import MahalanobisOOD
from .adapter import ensure_batched_scores

class AbstainityPipeline:
    def __init__(self, config_path: str = None):
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)
                
        # Initialize sub-layers
        self.scaler = TemperatureScaler(self.config.get("temperature", 1.0))
        self.mc = MCDropout(passes=10)
        
        # OOD (will safely handle None inputs internally if config lacks them)
        self.ood = MahalanobisOOD(
            class_means=self.config.get("class_means"),
            covariance_inv=self.config.get("covariance_inv")
        )
        
    def predict(self, adapter, model, x):
        """
        Executes the Abstainity pipeline. Matches JSON schema for the response.
        """
        layers_run = []
        tier = 1
        
        # Adapter implementations may return either [classes] or [batch, classes].
        # Normalize once at the pipeline boundary before applying dim=1 operations.
        model.eval()
        scores = adapter.predict(model, x)
        scores = ensure_batched_scores(scores)
        is_prob = (scores.min() >= 0.0) and torch.allclose(
            scores.sum(dim=1), torch.ones(scores.size(0), device=scores.device), atol=1e-2
        )
        if is_prob:
            raw_probs = scores
            raw_logits = torch.log(scores.clamp(min=1e-12))
        else:
            raw_logits = scores
            raw_probs = torch.softmax(raw_logits, dim=1)
            
        prediction = int(raw_probs.argmax(dim=1).item())
        raw_confidence = float(raw_probs.max(dim=1)[0].item())

        if "temperature" in self.config:
            calibrated_probs = self.scaler.scale(raw_logits)
            calibrated_confidence = float(calibrated_probs.max(dim=1)[0].item())
            layers_run.append("temperature_scaling")
            tier = max(tier, 3)
        else:
            calibrated_confidence = raw_confidence
            
        # 3. Uncertainty
        # get_uncertainty expects the function to return probabilities or logits.
        # We will pass a lambda that forces probabilities.
        def prob_predict_fn(m, inp):
            sc = ensure_batched_scores(adapter.predict(m, inp))
            if (sc.min() >= 0.0) and torch.allclose(
                sc.sum(dim=1), torch.ones(sc.size(0), device=sc.device), atol=1e-2
            ):
                return sc
            return torch.softmax(sc, dim=1)
            
        mc_probs, uncertainty, mc_tier, mc_layers = self.mc.get_uncertainty(model, x, prob_predict_fn)
        layers_run.extend(mc_layers)
        tier = max(tier, mc_tier)
        
        # 4. OOD
        ood_score = None
        ood_threshold = self.config.get("ood_threshold")
        feature_layer = getattr(adapter, "FEATURE_LAYER", None)
        
        if feature_layer and "class_means" in self.config:
            # We have OOD fit
            try:
                handle, features = self.ood.hook_layer(model, feature_layer)
                with torch.no_grad():
                    _ = ensure_batched_scores(adapter.predict(model, x))
                handle.remove()
                
                if len(features) > 0:
                    feat_np = features[-1].cpu().numpy()[0]
                    ood_score = self.ood.compute_distance(feat_np)
                    if ood_score is not None:
                        layers_run.append("mahalanobis_ood")
                        tier = max(tier, 3)
            except Exception:
                pass
                    
        # 5. Triage
        verdict, reason = self._triage(calibrated_confidence, uncertainty, ood_score, ood_threshold)
        
        return {
            "prediction": prediction,
            "raw_confidence": raw_confidence,
            "calibrated_confidence": calibrated_confidence,
            "uncertainty": uncertainty,
            "ood_score": ood_score,
            "ood_threshold": ood_threshold,
            "verdict": verdict,
            "reason": reason,
            "tier": tier,
            "layers_run": layers_run
        }
        
    def _get_percentile(self, value, sorted_ref):
        import bisect
        if not sorted_ref: return 0.0
        idx = bisect.bisect_left(sorted_ref, value)
        return idx / len(sorted_ref)

    def _triage(self, conf, uncert, ood, ood_thresh):
        if ood is not None and ood_thresh is not None and ood > ood_thresh:
            return "ABSTAIN", f"Out of distribution feature distance ({ood:.2f}) exceeded threshold ({ood_thresh:.2f})."
            
        if "ref_inv_conf" in self.config and "accept_risk_threshold" in self.config:
            conf_risk = self._get_percentile(1.0 - conf, self.config["ref_inv_conf"])
            uncert_risk = self._get_percentile(uncert, self.config["ref_uncert"])
            ood_risk = self._get_percentile(ood, self.config["ref_ood"]) if ood is not None else 0.0
            
            total_risk = (conf_risk + uncert_risk + ood_risk) / 3.0
            
            if total_risk <= self.config["accept_risk_threshold"]:
                return "ACCEPT", f"Composite risk ({total_risk:.2f}) meets ACCEPT threshold."
            elif total_risk <= self.config["uncertain_risk_threshold"]:
                return "UNCERTAIN", f"Composite risk ({total_risk:.2f}) in UNCERTAIN tier."
            else:
                return "ABSTAIN", f"Composite risk ({total_risk:.2f}) exceeds safe thresholds."

        # Fallback heuristics
        accept_conf = self.config.get("accept_conf_threshold", 0.8)
        reject_uncert = self.config.get("reject_uncert_threshold", 0.05)
        
        if uncert > reject_uncert:
            return "UNCERTAIN", f"High epistemic uncertainty ({uncert:.4f})."
        if conf < accept_conf:
            return "UNCERTAIN", f"Low calibrated confidence ({conf:.4f})."
            
        return "ACCEPT", "High confidence, low uncertainty, in-distribution."
