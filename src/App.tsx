import React, { useState, useRef } from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

export default function App() {
  const [step, setStep] = useState(1);
  const [logs, setLogs] = useState<string[]>([]);
  const [fitResult, setFitResult] = useState<any>(null);
  const [predictResult, setPredictResult] = useState<any>(null);
  const [testImagePreview, setTestImagePreview] = useState<string | null>(null);

  // Step 1 State
  const [modelType, setModelType] = useState('full_model');
  const [preset, setPreset] = useState('imagenet');
  const [modelClasses, setModelClasses] = useState<any[]>([]);
  const [selectedClass, setSelectedClass] = useState('');
  const [classArgs, setClassArgs] = useState<any>({});
  const [customSize, setCustomSize] = useState('(224, 224)');
  const [customMean, setCustomMean] = useState('[0.5, 0.5, 0.5]');
  const [customStd, setCustomStd] = useState('[0.5, 0.5, 0.5]');
  const classNamesRef = useRef<HTMLInputElement>(null);
  
  const modelFileRef = useRef<HTMLInputElement>(null);
  const scriptFileRef = useRef<HTMLInputElement>(null);
  const refZipRef = useRef<HTMLInputElement>(null);
  const testImgRef = useRef<HTMLInputElement>(null);

  const log = (msg: string) => setLogs((prev) => [...prev, msg]);

  const handleInspectScript = async (e: any) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('script', file);
    log('Inspecting model script for architectures...');
    const res = await fetch(`${API_URL}/inspect_model`, { method: 'POST', body: formData });
    if (res.ok) {
      const data = await res.json();
      setModelClasses(data.classes);
      if (data.classes.length > 0) {
        setSelectedClass(data.classes[0].name);
        const args = {};
        data.classes[0].args.forEach((a: string) => args[a] = '');
        setClassArgs(args);
      }
      log(`Found ${data.classes.length} architecture(s).`);
    } else {
      log('Failed to inspect model script.');
    }
  };

  const handleClassSelect = (clsName: string) => {
    setSelectedClass(clsName);
    const cls = modelClasses.find(c => c.name === clsName);
    const args = {};
    if (cls) {
      cls.args.forEach((a: string) => args[a] = '');
    }
    setClassArgs(args);
  };

  const handleUploadModel = async () => {
    const modelFile = modelFileRef.current?.files?.[0];
    if (!modelFile) return alert('Select a PyTorch model file');
    
    const formData = new FormData();
    formData.append('model_file', modelFile);
    formData.append('model_type', modelType);
    formData.append('preset', preset);
    
    if (modelType === 'weights_and_script') {
        const scriptFile = scriptFileRef.current?.files?.[0];
        if (!scriptFile) return alert('Select the architecture script file');
        formData.append('script_file', scriptFile);
        formData.append('class_name', selectedClass);
        formData.append('kwargs_json', JSON.stringify(classArgs));
    }
    
    if (preset === 'custom') {
        formData.append('custom_size', customSize);
        formData.append('custom_mean', customMean);
        formData.append('custom_std', customStd);
    }
    
    const classNamesFile = classNamesRef.current?.files?.[0];
    if (classNamesFile) {
        formData.append('class_names_file', classNamesFile);
    }

    log('Uploading model and generating internal configuration...');
    const res = await fetch(`${API_URL}/upload/model`, { method: 'POST', body: formData });
    if (res.ok) {
      log('Model uploaded and ready.');
      setStep(2);
    } else {
      log('Error uploading model.');
    }
  };

  const handleUploadRef = async () => {
    const file = refZipRef.current?.files?.[0];
    if (!file) return alert('Select reference.zip');
    const formData = new FormData();
    formData.append('file', file);

    log('Uploading reference data...');
    const res = await fetch(`${API_URL}/upload/reference`, { method: 'POST', body: formData });
    if (res.ok) {
      log('Reference data uploaded and extracted.');
      setStep(3);
    } else {
      log('Error uploading reference data.');
    }
  };

  const handleFit = async () => {
    log('Starting Abstainity fitting pipeline...');
    log('Loading model...');
    log('Splitting reference set...');
    log('Running Temperature calibration...');
    log('Computing feature statistics...');
    log('Calculating OOD and risk thresholds...');

    const res = await fetch(`${API_URL}/fit`, { method: 'POST' });
    if (res.ok) {
      const data = await res.json();
      setFitResult(data);
      log('Fitting complete! Ready for testing.');
      setStep(4);
    } else {
      log('Error during fitting.');
    }
  };

  const handleTest = async () => {
    const file = testImgRef.current?.files?.[0];
    if (!file) return alert('Select a test image');

    setTestImagePreview(URL.createObjectURL(file));

    const formData = new FormData();
    formData.append('image', file);

    log('Running Abstainity prediction...');
    const res = await fetch(`${API_URL}/predict`, { method: 'POST', body: formData });
    if (res.ok) {
      const data = await res.json();
      setPredictResult(data);
      setStep(5);
    } else {
      log('Error during prediction.');
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 p-8">
      <div className="max-w-5xl mx-auto space-y-8">
        
        {/* Header */}
        <header className="border-b pb-4">
          <h1 className="text-4xl font-bold tracking-tight text-slate-900">Abstainity</h1>
          <p className="text-xl text-slate-600 mt-2">Give your AI model the ability to say: I don't know.</p>
          <p className="text-sm text-slate-500 mt-1">A generic reliability layer for PyTorch image classifiers.</p>
        </header>

        {/* Wizard Flow */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          <div className="lg:col-span-1 space-y-6">
            
            {/* Step 1 */}
            <div className={`p-4 rounded-lg border ${step >= 1 ? 'bg-white border-blue-500 shadow-sm' : 'bg-slate-100 opacity-50'}`}>
              <h2 className="font-semibold mb-2">1. Bring Your PyTorch Model</h2>
              <p className="text-sm text-slate-600 mb-2">
                Upload your existing PyTorch model (.pt or .pth).
              </p>
              
              <div className="mb-3 space-y-2">
                  <label className="text-xs font-semibold block">Model Format</label>
                  <select 
                      className="w-full text-sm p-2 border rounded" 
                      value={modelType} 
                      onChange={(e) => setModelType(e.target.value)}
                      disabled={step !== 1}
                  >
                      <option value="full_model">Full Model / TorchScript (torch.save or torch.jit)</option>
                      <option value="weights_and_script">Weights State Dict + Architecture Script</option>
                      <option value="test_script">Auto-configure from plain test.py</option>
                  </select>
              </div>

              <div className="mb-3">
                  <label className="text-xs font-semibold block">Model File (.pt / .pth) [Optional for test.py]</label>
                  <input type="file" ref={modelFileRef} accept=".pt,.pth" className="text-sm w-full border p-1 rounded" disabled={step !== 1} />
              </div>

              {modelType === 'test_script' && (
                  <div className="mb-3 p-3 bg-slate-50 border rounded space-y-3">
                      <div>
                          <label className="text-xs font-semibold block">Inference Script (test.py)</label>
                          <input type="file" ref={scriptFileRef} accept=".py" className="text-sm w-full border p-1 rounded" disabled={step !== 1} />
                      </div>
                      <p className="text-xs text-slate-500 italic">Abstainity will try to automatically extract the model and preprocessing from your script.</p>
                  </div>
              )}

              {modelType === 'weights_and_script' && (
                  <div className="mb-3 p-3 bg-slate-50 border rounded space-y-3">
                      <div>
                          <label className="text-xs font-semibold block">Architecture Script (.py)</label>
                          <input type="file" ref={scriptFileRef} accept=".py" onChange={handleInspectScript} className="text-sm w-full border p-1 rounded" disabled={step !== 1} />
                      </div>
                      
                      {modelClasses.length > 0 && (
                          <>
                              <div>
                                  <label className="text-xs font-semibold block">Detected Class</label>
                                  <select 
                                      className="w-full text-sm p-1 border rounded" 
                                      value={selectedClass} 
                                      onChange={(e) => handleClassSelect(e.target.value)}
                                      disabled={step !== 1}
                                  >
                                      {modelClasses.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
                                  </select>
                              </div>
                              {Object.keys(classArgs).map(arg => (
                                  <div key={arg}>
                                      <label className="text-xs font-semibold block">Arg: {arg}</label>
                                      <input 
                                        type="text" 
                                        className="text-sm w-full border p-1 rounded" 
                                        value={classArgs[arg]} 
                                        onChange={(e) => setClassArgs({...classArgs, [arg]: e.target.value})} 
                                        disabled={step !== 1} 
                                      />
                                  </div>
                              ))}
                          </>
                      )}
                  </div>
              )}

              <div className="mb-4 space-y-2">
                  <label className="text-xs font-semibold block">Preprocessing Preset</label>
                  <select 
                      className="w-full text-sm p-2 border rounded" 
                      value={preset} 
                      onChange={(e) => setPreset(e.target.value)}
                      disabled={step !== 1}
                  >
                      <option value="imagenet">ImageNet (224x224 RGB)</option>
                      <option value="mnist">MNIST (28x28 Grayscale)</option>
                      <option value="clinsure">ClinSure X-Ray (224x224 RGB Custom)</option>
                      <option value="custom">Custom (Size, Mean, Std)</option>
                  </select>
              </div>

              {preset === 'custom' && (
                  <div className="mb-4 p-3 bg-slate-50 border rounded space-y-3">
                      <div>
                          <label className="text-xs font-semibold block">Image Size (e.g., (224, 224))</label>
                          <input type="text" value={customSize} onChange={(e) => setCustomSize(e.target.value)} className="text-sm w-full border p-1 rounded" disabled={step !== 1} />
                      </div>
                      <div>
                          <label className="text-xs font-semibold block">Mean (e.g., [0.5, 0.5, 0.5])</label>
                          <input type="text" value={customMean} onChange={(e) => setCustomMean(e.target.value)} className="text-sm w-full border p-1 rounded" disabled={step !== 1} />
                      </div>
                      <div>
                          <label className="text-xs font-semibold block">Std (e.g., [0.5, 0.5, 0.5])</label>
                          <input type="text" value={customStd} onChange={(e) => setCustomStd(e.target.value)} className="text-sm w-full border p-1 rounded" disabled={step !== 1} />
                      </div>
                  </div>
              )}
              
              <div className="mb-4">
                  <label className="text-xs font-semibold block">Class Names (.txt or .json) [Optional]</label>
                  <input type="file" ref={classNamesRef} accept=".txt,.json" className="text-sm w-full border p-1 rounded" disabled={step !== 1} />
              </div>

              {step === 1 && (
                <button onClick={handleUploadModel} className="px-4 py-2 bg-blue-600 text-white rounded text-sm w-full font-medium">Configure Model</button>
              )}
            </div>

            {/* Step 2 */}
            <div className={`p-4 rounded-lg border ${step >= 2 ? 'bg-white border-blue-500 shadow-sm' : 'bg-slate-100 opacity-50'}`}>
              <h2 className="font-semibold mb-2">2. Reference Data (Optional)</h2>
              <p className="text-sm text-slate-600 mb-2">
                Optionally provide representative labeled images from the model's normal operating domain (.zip). 
                Without this, Abstainity provides basic uncertainty analysis (Tier 1/2). With it, Abstainity additionally calibrates confidence and establishes OOD thresholds (Tier 3).
              </p>
              <p className="text-xs text-slate-500 mb-3 italic">
                Recommendation: ~50–100 images per class from your validation set.
              </p>
              <input type="file" ref={refZipRef} accept=".zip" className="text-sm mb-2 w-full" disabled={step !== 2} />
              {step === 2 && (
                <div className="flex gap-2">
                  <button onClick={handleUploadRef} className="px-4 py-2 bg-blue-600 text-white rounded text-sm w-full font-medium">Upload Dataset</button>
                  <button onClick={() => setStep(4)} className="px-4 py-2 bg-slate-200 text-slate-700 rounded text-sm w-full font-medium">Skip (Tier 1/2)</button>
                </div>
              )}
            </div>

            {/* Step 3 */}
            <div className={`p-4 rounded-lg border ${step === 3 ? 'bg-white border-blue-500 shadow-sm' : 'bg-slate-100 opacity-50'}`}>
              <h2 className="font-semibold mb-2">3. Fit Abstainity</h2>
              <p className="text-sm text-slate-600 mb-2">
                Abstainity learns how reliable predictions look for this model using the reference dataset.
              </p>
              {step === 3 && (
                <button onClick={handleFit} className="px-4 py-2 bg-blue-600 text-white rounded text-sm w-full font-medium">Run Fitting Pipeline</button>
              )}
              {step > 3 && step !== 4.5 && <p className="text-sm text-green-600 font-medium">✓ Fitted Successfully</p>}
            </div>

            {/* Step 4 */}
            <div className={`p-4 rounded-lg border ${step >= 4 ? 'bg-white border-blue-500 shadow-sm' : 'bg-slate-100 opacity-50'}`}>
              <h2 className="font-semibold mb-2">4. Test a New Image</h2>
              <p className="text-sm text-slate-600 mb-2">
                Test a new image and see whether the model should be trusted.
              </p>
              <input type="file" ref={testImgRef} className="text-sm mb-2 w-full" disabled={step < 4} />
              {step >= 4 && (
                <button onClick={handleTest} className="px-4 py-2 bg-blue-600 text-white rounded text-sm w-full font-medium">Test Model</button>
              )}
            </div>

          </div>

          <div className="lg:col-span-2 space-y-6">
            
            {/* Console / Status */}
            <div className="bg-slate-900 text-green-400 font-mono text-sm p-4 rounded-lg h-48 overflow-y-auto">
              {logs.length === 0 && <span className="text-slate-500">System ready... Waiting for model upload.</span>}
              {logs.map((l, i) => (
                <div key={i}>{l}</div>
              ))}
            </div>

            {/* Results Screen */}
            {step === 5 && predictResult && (
              <div className="bg-white p-6 rounded-lg border shadow-sm space-y-6">
                
                <div className="flex items-start gap-6">
                  {testImagePreview && (
                    <img src={testImagePreview} alt="Test Input" className="w-48 h-48 object-cover rounded-md border" />
                  )}
                  <div>
                    <h2 className="text-2xl font-bold mb-2">Analysis Result</h2>
                    <p className="text-slate-600 mb-4 italic">
                      The classifier made the prediction. Abstainity decides how much you should trust it.
                    </p>
                    <div className="inline-block px-3 py-1 rounded bg-slate-100 text-sm font-medium text-slate-700">
                      Ran on Capability Tier {predictResult.tier} ({predictResult.layers_run?.join(', ') || 'N/A'})
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-6">
                  {/* RAW MODEL */}
                  <div className="p-4 rounded-lg border bg-slate-50">
                    <h3 className="text-lg font-bold text-slate-700 mb-4">Raw Model Output</h3>
                    <div className="space-y-3">
                      <div>
                        <div className="text-sm text-slate-500">Prediction</div>
                        <div className="text-2xl font-bold">{predictResult.prediction}</div>
                      </div>
                      <div>
                        <div className="text-sm text-slate-500">Raw Confidence</div>
                        <div className="text-xl">{(predictResult.raw_confidence * 100).toFixed(1)}%</div>
                      </div>
                      <div className="pt-4 border-t border-slate-200">
                        <p className="text-sm text-slate-500 italic">
                          Standard models are often overconfident on incorrect or out-of-distribution inputs.
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* ABSTAINITY */}
                  <div className={`p-4 rounded-lg border-2 ${
                    predictResult.verdict === 'ACCEPT' ? 'border-green-500 bg-green-50' : 
                    predictResult.verdict === 'UNCERTAIN' ? 'border-yellow-500 bg-yellow-50' : 
                    'border-red-500 bg-red-50'
                  }`}>
                    <h3 className="text-lg font-bold text-slate-900 mb-4">Abstainity Layer</h3>
                    <div className="space-y-3">
                      <div>
                        <div className="text-sm font-medium opacity-80">Verdict</div>
                        <div className={`text-2xl font-black ${
                          predictResult.verdict === 'ACCEPT' ? 'text-green-700' : 
                          predictResult.verdict === 'UNCERTAIN' ? 'text-yellow-700' : 
                          'text-red-700'
                        }`}>
                          {predictResult.verdict}
                        </div>
                      </div>
                      
                      {predictResult.tier === 3 ? (
                        <div>
                          <div className="text-sm font-medium opacity-80">Calibrated Confidence</div>
                          <div className="text-lg">{(predictResult.calibrated_confidence * 100).toFixed(1)}%</div>
                        </div>
                      ) : null}

                      <div>
                        <div className="text-sm font-medium opacity-80">Epistemic Uncertainty</div>
                        <div className="text-lg">{predictResult.uncertainty.toFixed(4)}</div>
                      </div>

                      {predictResult.tier === 3 ? (
                        <div>
                          <div className="text-sm font-medium opacity-80">Mahalanobis OOD Score</div>
                          <div className="text-lg">{predictResult.ood_score ? predictResult.ood_score.toFixed(2) : 'N/A'}</div>
                        </div>
                      ) : null}

                      <div className="pt-4 border-t border-slate-300">
                        <div className="text-sm font-medium opacity-80">Reason</div>
                        <p className="text-sm font-semibold">{predictResult.reason}</p>
                      </div>
                    </div>
                  </div>

                </div>

                {/* Contextual Explanations for This Image */}
                <div className="mt-6 border-t pt-6">
                  <h3 className="text-lg font-bold text-slate-900 mb-4">What Abstainity Saw For This Image</h3>
                  <div className="space-y-4">
                    
                    {/* Calibration Explanation */}
                    <div className="flex gap-4 items-start">
                      <div className="mt-1 w-8 h-8 shrink-0 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 font-bold text-sm">1</div>
                      <div>
                        <h4 className="font-semibold text-slate-800">Calibration</h4>
                        {predictResult.tier === 3 ? (
                          <p className="text-sm text-slate-600">
                            The model output a raw confidence of <strong>{(predictResult.raw_confidence * 100).toFixed(1)}%</strong>. 
                            Abstainity's temperature scaler mathematically adjusted this to <strong>{(predictResult.calibrated_confidence * 100).toFixed(1)}%</strong> 
                            based on the reference domain's overconfidence profile.
                          </p>
                        ) : (
                          <p className="text-sm text-slate-500 italic">
                            Skipped. No reference dataset was provided in Step 2 to fit the temperature scaler.
                          </p>
                        )}
                      </div>
                    </div>

                    {/* Uncertainty Explanation */}
                    <div className="flex gap-4 items-start">
                      <div className="mt-1 w-8 h-8 shrink-0 rounded-full bg-yellow-100 flex items-center justify-center text-yellow-600 font-bold text-sm">2</div>
                      <div>
                        <h4 className="font-semibold text-slate-800">Epistemic Uncertainty</h4>
                        <p className="text-sm text-slate-600">
                          Across stochastic forward passes, the prediction variance was <strong>{predictResult.uncertainty.toFixed(4)}</strong>. 
                          {predictResult.uncertainty < 0.01 ? " This indicates the model's prediction was highly stable and not guessing." : 
                           predictResult.uncertainty < 0.05 ? " This indicates mild instability in the prediction." : 
                           " This indicates severe instability; the model is guessing."}
                        </p>
                      </div>
                    </div>

                    {/* OOD Explanation */}
                    <div className="flex gap-4 items-start">
                      <div className="mt-1 w-8 h-8 shrink-0 rounded-full bg-red-100 flex items-center justify-center text-red-600 font-bold text-sm">3</div>
                      <div>
                        <h4 className="font-semibold text-slate-800">OOD Detection (Mahalanobis)</h4>
                        {predictResult.tier === 3 ? (
                          <p className="text-sm text-slate-600">
                            This image's deep feature extraction produced a Mahalanobis distance of <strong>{predictResult.ood_score ? predictResult.ood_score.toFixed(2) : 'N/A'}</strong> from the normal operating domain. 
                            {predictResult.ood_threshold && predictResult.ood_score > predictResult.ood_threshold 
                              ? ` This exceeds the safe threshold of ${predictResult.ood_threshold.toFixed(2)}, meaning this image is an alien/out-of-distribution input!`
                              : ` This is safely within the established threshold of ${predictResult.ood_threshold ? predictResult.ood_threshold.toFixed(2) : 'N/A'}.`}
                          </p>
                        ) : (
                          <p className="text-sm text-slate-500 italic">
                            Skipped. No reference dataset was provided in Step 2 to map the normal operating domain.
                          </p>
                        )}
                      </div>
                    </div>

                  </div>
                </div>

              </div>
            )}

          </div>
        </div>

      </div>
    </div>
  );
}
