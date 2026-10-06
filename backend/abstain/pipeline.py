import torch
import json
import os
from .layers.calibration import TemperatureScaler
from .layers.mc_dropout import MCDropout
from .layers.ood import MahalanobisOOD

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
        
        # 1. Raw Prediction
        model.eval()
        with torch.no_grad():
            raw_probs = adapter.predict(model, x)
            prediction = int(raw_probs.argmax(dim=1).item())
            raw_confidence = float(raw_probs.max(dim=1)[0].item())
            
        # 2. Calibration
        # We only consider tier 3 if we actually have fitted temperature in config
        if "temperature" in self.config:
            # Recompute calibrated probabilities from raw logits? 
            # Wait, adapter.predict returns probabilities. Temperature scaling requires logits.
            # If the user script returns probabilities, we can't do temperature scaling directly unless they expose logits.
            # Let's assume adapter.predict returns logits for Abstainity if we want T-scaling, 
            # or we skip T-scaling if we only have probs.
            # For now, let's assume predict returns LOGITS if Phase 2 is used, or we just do our best.
            # Actually, standard contract: adapter.predict(model, x) -> probabilities (from my schema).
            # If so, T-scaling is impossible. Let's fix the adapter contract to return logits. 
            pass # We will fix this in a moment. Let's assume `raw_probs` are actually probabilities and we skip calibration if not possible, or we enforce logits.
            
        # Let's assume `adapter.predict` returns probabilities as per the adapter.py definition.
        # So we can't do temperature scaling without logits. I will modify adapter to return logits if possible,
        # but for now, calibrated_confidence = raw_confidence if we can't scale.
        # Let's assume we update adapter contract to return LOGITS to fully support Phase 1.
        # If it returns logits:
        # raw_logits = adapter.predict(model, x)
        # raw_probs = torch.softmax(raw_logits, dim=1)
        # We will assume adapter.predict returns LOGITS from now on, as it's the only way T-scaling works.
        # I'll update the adapter.py next.
        
        raw_logits = adapter.predict(model, x)
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
            return torch.softmax(adapter.predict(m, inp), dim=1)
            
        mc_probs, uncertainty, mc_tier, mc_layers = self.mc.get_uncertainty(model, x, prob_predict_fn)
        layers_run.extend(mc_layers)
        tier = max(tier, mc_tier)
        
        # 4. OOD
        ood_score = None
        ood_threshold = self.config.get("ood_threshold")
        feature_layer = getattr(adapter, "FEATURE_LAYER", None)
        
        if feature_layer and "class_means" in self.config:
            # We have OOD fit
            handle, features = self.ood.hook_layer(model, feature_layer)
            with torch.no_grad():
                _ = adapter.predict(model, x)
            handle.remove()
            
            if len(features) > 0:
                feat_np = features[0].cpu().numpy()[0]
                ood_score = self.ood.compute_distance(feat_np)
                if ood_score is not None:
                    layers_run.append("mahalanobis_ood")
                    tier = max(tier, 3)
                    
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
