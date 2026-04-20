import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

type JobStatus = {
  job_id: string
  status: string
  progress: number
  stage: string
  metrics: Record<string, unknown>
  plots: string[]
  has_results_zip: boolean
  has_predictions_zip: boolean
  error: string | null
  config: Record<string, unknown>
}

type Step = 'templates' | 'upload' | 'config' | 'running' | 'results' | 'prediction'

function App() {
  const [step, setStep] = useState<Step>('templates')
  const [trainingFile, setTrainingFile] = useState<File | null>(null)
  const [trainingFileId, setTrainingFileId] = useState('')
  const [newDataFile, setNewDataFile] = useState<File | null>(null)
  const [newDataFileId, setNewDataFileId] = useState('')
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState('')
  const [templateLabels, setTemplateLabels] = useState(7)
  const [templateInputs, setTemplateInputs] = useState(10)
  const [templateOutputName, setTemplateOutputName] = useState('Output')

  // In-app training data table
  const [trainingTableCols, setTrainingTableCols] = useState<string[]>([])
  const [trainingTableRows, setTrainingTableRows] = useState<string[][]>([])
  const [trainingTableActive, setTrainingTableActive] = useState(false)

  // In-app prediction data table
  const [predictionTableCols, setPredictionTableCols] = useState<string[]>([])
  const [predictionTableRows, setPredictionTableRows] = useState<string[][]>([])
  const [predictionTableActive, setPredictionTableActive] = useState(false)

  // Data source mode
  const [dataSource, setDataSource] = useState<'table' | 'file'>('table')

  // Instructions modal
  const [showInstructions, setShowInstructions] = useState(false)
  const [showAdvice, setShowAdvice] = useState(false)

  const [config, setConfig] = useState({
    train_percent: 60,
    val_percent: 20,
    test_percent: 20,
    n_labels: 7,
    n_inputs: 10,
    sep: '\\t',
    tuner_trials: 10,
    k_folds: 10,
    random_seed: 42,
    tuner_epochs: 200,
    cv_epochs: 200,
    final_epochs: 200,
    do_optional_retrain: true,
    pi_calibration: 'val',
    pi_alpha: 0.05,
    disable_gpu: true,
  })

  const [jobId, setJobId] = useState('')
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [logSince, setLogSince] = useState(0)
  const [plots, setPlots] = useState<Record<string, string>>({})
  const [selectedPlotCategory, setSelectedPlotCategory] = useState('diagnostics')

  const logsEndRef = useRef<HTMLDivElement>(null)
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const uploadFile = async (file: File, fileType: string): Promise<string> => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('file_type', fileType)
    const res = await fetch(`${API_URL}/api/upload`, { method: 'POST', body: formData })
    if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`)
    const data = await res.json()
    return data.file_id
  }

  const uploadTableData = async (cols: string[], rows: string[][], categories: string[], fileType: string): Promise<string> => {
    const res = await fetch(`${API_URL}/api/upload-table`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_type: fileType, categories, columns: cols, rows }),
    })
    if (!res.ok) throw new Error(`Table upload failed: ${res.statusText}`)
    const data = await res.json()
    return data.file_id
  }

  const handleUpload = async () => {
    setUploading(true)
    setUploadError('')
    try {
      if (dataSource === 'table' && trainingTableActive) {
        const categories = buildCategories(templateLabels, templateInputs, true)
        const fid = await uploadTableData(trainingTableCols, trainingTableRows, categories, 'training')
        setTrainingFileId(fid)
      } else if (trainingFile) {
        const fid = await uploadFile(trainingFile, 'training')
        setTrainingFileId(fid)
      } else {
        setUploadError('No training data provided. Enter data in the table or upload a file.')
        setUploading(false)
        return
      }
      if (newDataFile) {
        const nfid = await uploadFile(newDataFile, 'new_data')
        setNewDataFileId(nfid)
      }
      setStep('config')
    } catch (e) {
      setUploadError(e instanceof Error ? e.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  const buildCategories = (nLabels: number, nInputs: number, includeOutput: boolean): string[] => {
    const cats: string[] = []
    for (let i = 0; i < nLabels; i++) cats.push('Label')
    for (let i = 0; i < nInputs; i++) cats.push('Input')
    if (includeOutput) cats.push('Output')
    return cats
  }

  const initTrainingTable = () => {
    const totalCols = templateLabels + templateInputs + 1
    const cols: string[] = []
    for (let i = 0; i < templateLabels; i++) cols.push(`Label_${i + 1}`)
    for (let i = 0; i < templateInputs; i++) cols.push(`Input_${i + 1}`)
    cols.push(templateOutputName || 'Output')
    const rows: string[][] = Array.from({ length: 5 }, () => Array(totalCols).fill(''))
    setTrainingTableCols(cols)
    setTrainingTableRows(rows)
    setTrainingTableActive(true)
  }

  const initPredictionTable = () => {
    const totalCols = templateLabels + templateInputs
    const cols: string[] = []
    for (let i = 0; i < templateLabels; i++) cols.push(`Label_${i + 1}`)
    for (let i = 0; i < templateInputs; i++) cols.push(`Input_${i + 1}`)
    const rows: string[][] = Array.from({ length: 5 }, () => Array(totalCols).fill(''))
    setPredictionTableCols(cols)
    setPredictionTableRows(rows)
    setPredictionTableActive(true)
  }

  const updateTableCell = (rows: string[][], setRows: (r: string[][]) => void, rowIdx: number, colIdx: number, value: string) => {
    const newRows = rows.map((r, ri) => ri === rowIdx ? r.map((c, ci) => ci === colIdx ? value : c) : [...r])
    setRows(newRows)
  }

  const updateColumnName = (cols: string[], setCols: (c: string[]) => void, colIdx: number, value: string) => {
    const newCols = [...cols]
    newCols[colIdx] = value
    setCols(newCols)
  }

  const addRow = (rows: string[][], setRows: (r: string[][]) => void, nCols: number) => {
    setRows([...rows, Array(nCols).fill('')])
  }

  const removeRow = (rows: string[][], setRows: (r: string[][]) => void, rowIdx: number) => {
    if (rows.length <= 1) return
    setRows(rows.filter((_, i) => i !== rowIdx))
  }

  const handleTablePaste = (
    e: React.ClipboardEvent<HTMLInputElement>,
    rows: string[][],
    setRows: (r: string[][]) => void,
    rowIdx: number,
    colIdx: number,
    nCols: number,
  ) => {
    const text = e.clipboardData.getData('text/plain')
    // Check if pasted text contains tabs or newlines (multi-cell Excel paste)
    if (text.includes('\t') || text.includes('\n')) {
      e.preventDefault()
      const pastedRows = text.split(/\r?\n/).filter(line => line.length > 0)
      const newRows = rows.map(r => [...r])
      // Extend rows if needed
      while (newRows.length < rowIdx + pastedRows.length) {
        newRows.push(Array(nCols).fill(''))
      }
      for (let ri = 0; ri < pastedRows.length; ri++) {
        const cells = pastedRows[ri].split('\t')
        for (let ci = 0; ci < cells.length; ci++) {
          const targetCol = colIdx + ci
          const targetRow = rowIdx + ri
          if (targetCol < nCols && targetRow < newRows.length) {
            newRows[targetRow][targetCol] = cells[ci]
          }
        }
      }
      setRows(newRows)
    }
  }

  const startTraining = async () => {
    const formData = new FormData()
    formData.append('data_file_id', trainingFileId)
    if (newDataFileId) formData.append('new_data_file_id', newDataFileId)
    const cfgToSend = { ...config, sep: config.sep === '\\t' ? '\t' : config.sep }
    formData.append('config', JSON.stringify(cfgToSend))
    const res = await fetch(`${API_URL}/api/train`, { method: 'POST', body: formData })
    if (!res.ok) throw new Error('Failed to start training')
    const data = await res.json()
    setJobId(data.job_id)
    setStep('running')
    setLogs([])
    setLogSince(0)
    setPlots({})
  }

  const pollJob = useCallback(async () => {
    if (!jobId) return
    try {
      const [statusRes, logsRes] = await Promise.all([
        fetch(`${API_URL}/api/job/${jobId}`),
        fetch(`${API_URL}/api/job/${jobId}/logs?since=${logSince}`),
      ])
      if (statusRes.ok) {
        const status: JobStatus = await statusRes.json()
        setJobStatus(status)
        if (status.status === 'completed' || status.status === 'failed') {
          if (pollingRef.current) {
            clearInterval(pollingRef.current)
            pollingRef.current = null
          }
          if (status.status === 'completed') {
            setStep('results')
            for (const plotName of status.plots) {
              fetch(`${API_URL}/api/job/${jobId}/plot/${plotName}`)
                .then(r => r.json())
                .then(d => setPlots(prev => ({ ...prev, [plotName]: d.data })))
                .catch(() => {})
            }
          }
        }
      }
      if (logsRes.ok) {
        const logsData = await logsRes.json()
        if (logsData.logs.length > 0) {
          setLogs(prev => [...prev, ...logsData.logs])
          setLogSince(logsData.total)
        }
      }
    } catch {
      // ignore polling errors
    }
  }, [jobId, logSince])

  useEffect(() => {
    if (step === 'running' && jobId) {
      pollingRef.current = setInterval(pollJob, 2000)
      pollJob()
      return () => { if (pollingRef.current) clearInterval(pollingRef.current) }
    }
  }, [step, jobId, pollJob])

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  useEffect(() => {
    if (step === 'results' && jobStatus && jobStatus.plots.length > 0) {
      for (const plotName of jobStatus.plots) {
        if (!plots[plotName]) {
          fetch(`${API_URL}/api/job/${jobId}/plot/${plotName}`)
            .then(r => r.json())
            .then(d => setPlots(prev => ({ ...prev, [plotName]: d.data })))
            .catch(() => {})
        }
      }
    }
  }, [step, jobStatus, jobId, plots])

  const updateConfig = (key: string, value: unknown) => {
    setConfig(prev => ({ ...prev, [key]: value }))
  }

  const fmtVal = (v: unknown): string => {
    if (typeof v === 'number') return Number.isInteger(v) ? v.toString() : v.toFixed(4)
    if (typeof v === 'object' && v !== null) return JSON.stringify(v)
    return String(v)
  }

  const catPlots = {
    diagnostics: (jobStatus?.plots || []).filter(p => p.includes('pred_vs_actual') || p.includes('mse_evolution')),
    residuals: (jobStatus?.plots || []).filter(p => p.includes('residual')),
    surfaces: (jobStatus?.plots || []).filter(p => p.includes('surface')),
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            {/* ANN Scheme Diagram */}
            <svg width="64" height="56" viewBox="0 0 120 100" className="flex-shrink-0">
              {/* Connections: Input -> Hidden1 */}
              <line x1="20" y1="20" x2="50" y2="15" stroke="#93c5fd" strokeWidth="1" />
              <line x1="20" y1="20" x2="50" y2="35" stroke="#93c5fd" strokeWidth="1" />
              <line x1="20" y1="20" x2="50" y2="55" stroke="#93c5fd" strokeWidth="1" />
              <line x1="20" y1="20" x2="50" y2="75" stroke="#93c5fd" strokeWidth="1" />
              <line x1="20" y1="50" x2="50" y2="15" stroke="#93c5fd" strokeWidth="1" />
              <line x1="20" y1="50" x2="50" y2="35" stroke="#93c5fd" strokeWidth="1" />
              <line x1="20" y1="50" x2="50" y2="55" stroke="#93c5fd" strokeWidth="1" />
              <line x1="20" y1="50" x2="50" y2="75" stroke="#93c5fd" strokeWidth="1" />
              <line x1="20" y1="80" x2="50" y2="15" stroke="#93c5fd" strokeWidth="1" />
              <line x1="20" y1="80" x2="50" y2="35" stroke="#93c5fd" strokeWidth="1" />
              <line x1="20" y1="80" x2="50" y2="55" stroke="#93c5fd" strokeWidth="1" />
              <line x1="20" y1="80" x2="50" y2="75" stroke="#93c5fd" strokeWidth="1" />
              {/* Connections: Hidden1 -> Hidden2 */}
              <line x1="50" y1="15" x2="80" y2="25" stroke="#6ee7b7" strokeWidth="1" />
              <line x1="50" y1="15" x2="80" y2="50" stroke="#6ee7b7" strokeWidth="1" />
              <line x1="50" y1="15" x2="80" y2="75" stroke="#6ee7b7" strokeWidth="1" />
              <line x1="50" y1="35" x2="80" y2="25" stroke="#6ee7b7" strokeWidth="1" />
              <line x1="50" y1="35" x2="80" y2="50" stroke="#6ee7b7" strokeWidth="1" />
              <line x1="50" y1="35" x2="80" y2="75" stroke="#6ee7b7" strokeWidth="1" />
              <line x1="50" y1="55" x2="80" y2="25" stroke="#6ee7b7" strokeWidth="1" />
              <line x1="50" y1="55" x2="80" y2="50" stroke="#6ee7b7" strokeWidth="1" />
              <line x1="50" y1="55" x2="80" y2="75" stroke="#6ee7b7" strokeWidth="1" />
              <line x1="50" y1="75" x2="80" y2="25" stroke="#6ee7b7" strokeWidth="1" />
              <line x1="50" y1="75" x2="80" y2="50" stroke="#6ee7b7" strokeWidth="1" />
              <line x1="50" y1="75" x2="80" y2="75" stroke="#6ee7b7" strokeWidth="1" />
              {/* Connections: Hidden2 -> Output */}
              <line x1="80" y1="25" x2="108" y2="50" stroke="#fca5a5" strokeWidth="1" />
              <line x1="80" y1="50" x2="108" y2="50" stroke="#fca5a5" strokeWidth="1" />
              <line x1="80" y1="75" x2="108" y2="50" stroke="#fca5a5" strokeWidth="1" />
              {/* Input nodes */}
              <circle cx="20" cy="20" r="7" fill="#2563eb" />
              <circle cx="20" cy="50" r="7" fill="#2563eb" />
              <circle cx="20" cy="80" r="7" fill="#2563eb" />
              {/* Hidden layer 1 nodes */}
              <circle cx="50" cy="15" r="6" fill="#059669" />
              <circle cx="50" cy="35" r="6" fill="#059669" />
              <circle cx="50" cy="55" r="6" fill="#059669" />
              <circle cx="50" cy="75" r="6" fill="#059669" />
              {/* Hidden layer 2 nodes */}
              <circle cx="80" cy="25" r="6" fill="#059669" />
              <circle cx="80" cy="50" r="6" fill="#059669" />
              <circle cx="80" cy="75" r="6" fill="#059669" />
              {/* Output node */}
              <circle cx="108" cy="50" r="7" fill="#dc2626" />
              {/* Labels */}
              <text x="20" y="97" textAnchor="middle" fontSize="7"               fill="#4b5563">Input</text>
                            <text x="65" y="97" textAnchor="middle" fontSize="7" fill="#4b5563">Hidden</text>
                            <text x="108" y="97" textAnchor="middle" fontSize="7" fill="#4b5563">Output</text>
            </svg>
            <div>
              <h1 className="text-xl font-bold text-gray-900">ANNSIMS</h1>
              <p className="text-sm text-gray-800">Artificial Neural Networks Modelling for stable isotopes SIMS analyses</p>
            </div>
          </div>
          <div className="flex items-center gap-2 mr-4">
            <button onClick={() => setShowInstructions(true)}
              className="px-4 py-2 bg-blue-50 text-blue-700 text-sm font-medium rounded-lg border border-blue-200 hover:bg-blue-100 transition-colors flex items-center gap-2 whitespace-nowrap">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Instructions
            </button>
            <button onClick={() => setShowAdvice(true)}
              className="px-4 py-2 bg-amber-50 text-amber-700 text-sm font-medium rounded-lg border border-amber-200 hover:bg-amber-100 transition-colors flex items-center gap-2 whitespace-nowrap">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
              Modelling Advice
            </button>
          </div>
          <div className="flex items-center gap-1 flex-wrap justify-end">
            {(['templates', 'upload', 'config', 'running', 'results', 'prediction'] as Step[]).map((s, i) => {
              const allSteps: Step[] = ['templates', 'upload', 'config', 'running', 'results', 'prediction']
              const stepLabels: Record<Step, string> = {
                templates: 'Prepare Data Templates',
                upload: 'Training Data Upload',
                config: 'Training Configuration',
                running: 'Training Running',
                results: 'Training Results',
                prediction: 'Samples Prediction',
              }
              const currentIdx = allSteps.indexOf(step)
              return (
              <div key={s} className="flex items-center gap-1">
                <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium ${
                  step === s ? 'bg-blue-600 text-white' :
                  currentIdx > i ? 'bg-green-500 text-white' :
                  'bg-gray-200 text-gray-800'
                }`}>
                  {currentIdx > i ? '\u2713' : i + 1}
                </div>
                <span className={`text-xs ${step === s ? 'text-blue-600 font-medium' : 'text-gray-800'}`}>{stepLabels[s]}</span>
                {i < 5 && <div className="w-4 h-px bg-gray-300" />}
              </div>
              )
            })}
          </div>
        </div>
      </header>

      {/* Modelling Advice Modal */}
      {showAdvice && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl shadow-xl max-w-xl w-full">
            <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between rounded-t-2xl">
              <h2 className="text-xl font-bold text-gray-900">Modelling Advice</h2>
              <button onClick={() => setShowAdvice(false)} className="text-gray-800 hover:text-gray-600 text-2xl leading-none">&times;</button>
            </div>
            <div className="px-6 py-5 space-y-5">
              <div className="flex gap-3">
                <div className="w-7 h-7 rounded-full bg-amber-500 text-white flex items-center justify-center text-xs font-bold flex-shrink-0 mt-0.5">1</div>
                <p className="text-sm text-gray-800">Artificial Neural Networks are predictive models that rely on <strong>data quality</strong>.</p>
              </div>
              <div className="flex gap-3">
                <div className="w-7 h-7 rounded-full bg-amber-500 text-white flex items-center justify-center text-xs font-bold flex-shrink-0 mt-0.5">2</div>
                <p className="text-sm text-gray-800">Predictive models are <strong>not intended to return physical interpretations</strong>.</p>
              </div>
              <div className="flex gap-3">
                <div className="w-7 h-7 rounded-full bg-amber-500 text-white flex items-center justify-center text-xs font-bold flex-shrink-0 mt-0.5">3</div>
                <p className="text-sm text-gray-800">It is recommended that the ranges of values of the inputs of the training data <strong>cover the range of values of the inputs of the unknown samples</strong> that will be predicted to avoid extrapolation.</p>
              </div>
            </div>
            <div className="sticky bottom-0 bg-white border-t border-gray-200 px-6 py-4 rounded-b-2xl">
              <button onClick={() => setShowAdvice(false)}
                className="w-full py-2.5 bg-amber-500 text-white font-medium rounded-lg hover:bg-amber-600 transition-colors">
                Got it
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Instructions Modal */}
      {showInstructions && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl shadow-xl max-w-2xl w-full max-h-[85vh] overflow-y-auto">
            <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between rounded-t-2xl">
              <h2 className="text-xl font-bold text-gray-900">How to Use This Pipeline</h2>
              <button onClick={() => setShowInstructions(false)} className="text-gray-800 hover:text-gray-600 text-2xl leading-none">&times;</button>
            </div>
            <div className="px-6 py-5 space-y-5">
              <div className="flex gap-3">
                <div className="w-7 h-7 rounded-full bg-blue-600 text-white flex items-center justify-center text-xs font-bold flex-shrink-0 mt-0.5">1</div>
                <div>
                  <h3 className="font-semibold text-gray-900">Prepare Data Templates</h3>
                  <p className="text-sm text-gray-800 mt-1">Configure the number of <span className="text-blue-600 font-medium">Label</span> columns (identifiers/metadata), <span className="text-green-600 font-medium">Input</span> columns (model features), and the <span className="text-red-600 font-medium">Output</span> column (target variable). Then either:</p>
                  <ul className="text-sm text-gray-800 mt-1 ml-4 list-disc space-y-1">
                    <li><strong>Enter data directly</strong> in the in-app table — click "Create Table", type values or paste from Excel (Ctrl+V)</li>
                    <li><strong>Download a template</strong> (.xlsx) to fill externally and upload later</li>
                  </ul>
                </div>
              </div>
              <div className="flex gap-3">
                <div className="w-7 h-7 rounded-full bg-blue-600 text-white flex items-center justify-center text-xs font-bold flex-shrink-0 mt-0.5">2</div>
                <div>
                  <h3 className="font-semibold text-gray-900">Training Data Upload</h3>
                  <p className="text-sm text-gray-800 mt-1">Choose your data source: use the table data entered in Step 1, or upload a file (TSV, CSV, or Excel). The system auto-detects the template format (Row 1 = categories, Row 2 = column names, Row 3+ = data).</p>
                </div>
              </div>
              <div className="flex gap-3">
                <div className="w-7 h-7 rounded-full bg-blue-600 text-white flex items-center justify-center text-xs font-bold flex-shrink-0 mt-0.5">3</div>
                <div>
                  <h3 className="font-semibold text-gray-900">Training Configuration</h3>
                  <p className="text-sm text-gray-800 mt-1">Adjust pipeline parameters:</p>
                  <ul className="text-sm text-gray-800 mt-1 ml-4 list-disc space-y-1">
                    <li><strong>Data Split</strong> — Train/Validation/Test percentages</li>
                    <li><strong>Tuner Trials</strong> — number of hyperparameter search trials</li>
                    <li><strong>K-Folds</strong> — cross-validation folds</li>
                    <li><strong>Epochs</strong> — tuner, CV, and final training epochs</li>
                    <li><strong>PI Alpha</strong> — prediction interval confidence (0.05 = 95%)</li>
                  </ul>
                </div>
              </div>
              <div className="flex gap-3">
                <div className="w-7 h-7 rounded-full bg-blue-600 text-white flex items-center justify-center text-xs font-bold flex-shrink-0 mt-0.5">4</div>
                <div>
                  <h3 className="font-semibold text-gray-900">Training Running</h3>
                  <p className="text-sm text-gray-800 mt-1">The ANN pipeline runs through: hyperparameter tuning → K-fold cross-validation → final model training → optional retrain on Train+Val → diagnostics & plot generation. A live progress bar and log viewer show real-time updates.</p>
                </div>
              </div>
              <div className="flex gap-3">
                <div className="w-7 h-7 rounded-full bg-blue-600 text-white flex items-center justify-center text-xs font-bold flex-shrink-0 mt-0.5">5</div>
                <div>
                  <h3 className="font-semibold text-gray-900">Training Results</h3>
                  <p className="text-sm text-gray-800 mt-1">View model performance metrics (R², RMSE, MAE, MSE) for Train, Validation, and Test sets. Cross-validation statistics and best hyperparameters are also shown. Browse diagnostic plots (MSE evolution, predicted vs actual), residual plots, and 3D surface visualizations. Download all results as a ZIP file.</p>
                </div>
              </div>
              <div className="flex gap-3">
                <div className="w-7 h-7 rounded-full bg-blue-600 text-white flex items-center justify-center text-xs font-bold flex-shrink-0 mt-0.5">6</div>
                <div>
                  <h3 className="font-semibold text-gray-900">Samples Prediction</h3>
                  <p className="text-sm text-gray-800 mt-1">Upload new samples (without the Output column) to generate predictions using the trained model. You can enter data in the prediction table or upload a file. Input columns must match the training data structure.</p>
                </div>
              </div>
              <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                <h3 className="font-semibold text-gray-900 text-sm mb-2">Tips</h3>
                <ul className="text-sm text-gray-800 space-y-1 list-disc ml-4">
                  <li>You can <strong>copy/paste data from Excel</strong> directly into any data table — select cells in Excel, then Ctrl+V into any cell in the app</li>
                  <li>The Excel template uses a <strong>2-header-row format</strong>: Row 1 = category (Label/Input/Output), Row 2 = column names, Row 3+ = data</li>
                  <li>All plots are exported as <strong>TIFF format at 600 dpi</strong> in the downloadable ZIP</li>
                  <li>Set <strong>Label Columns to 0</strong> if your data has no identifier columns</li>
                </ul>
              </div>
            </div>
            <div className="sticky bottom-0 bg-white border-t border-gray-200 px-6 py-4 rounded-b-2xl">
              <button onClick={() => setShowInstructions(false)}
                className="w-full py-2.5 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition-colors">
                Got it
              </button>
            </div>
          </div>
        </div>
      )}

      <main className="max-w-7xl mx-auto px-4 py-8">
        {step === 'templates' && (
          <div className="max-w-6xl mx-auto space-y-6">
            {/* Column configuration */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-bold text-gray-900 mb-2">Configure Data Structure</h2>
              <p className="text-gray-800 text-sm mb-4">Set the number of columns for your data, then enter data directly or download templates.</p>
              <div className="grid grid-cols-3 gap-4 mb-4">
                <div>
                  <label className="block text-sm font-medium text-gray-800 mb-1">Label Columns</label>
                  <input type="number" min={0} max={50} value={templateLabels} onChange={e => setTemplateLabels(Number(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" />
                  <p className="text-xs text-gray-800 mt-1">Metadata / identifiers</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-800 mb-1">Input Columns</label>
                  <input type="number" min={1} max={100} value={templateInputs} onChange={e => setTemplateInputs(Number(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" />
                  <p className="text-xs text-gray-800 mt-1">Model features</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-800 mb-1">Output Column Name</label>
                  <input type="text" value={templateOutputName} onChange={e => setTemplateOutputName(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" />
                  <p className="text-xs text-gray-800 mt-1">Target variable</p>
                </div>
              </div>
            </div>

            {/* Training Data Table */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h2 className="text-lg font-bold text-gray-900">Training Data Table</h2>
                  <p className="text-gray-800 text-sm">Enter your training data directly. Column names are editable.</p>
                </div>
                <div className="flex gap-2">
                  <button onClick={initTrainingTable}
                    className="px-4 py-2 bg-emerald-600 text-white text-sm font-medium rounded-lg hover:bg-emerald-700 transition-colors">
                    {trainingTableActive ? 'Reset Table' : 'Create Table'}
                  </button>
                  <a href={`${API_URL}/api/template?n_labels=${templateLabels}&n_inputs=${templateInputs}&output_name=${encodeURIComponent(templateOutputName)}`}
                    className="px-4 py-2 bg-gray-100 text-gray-800 text-sm font-medium rounded-lg hover:bg-gray-200 transition-colors">
                    Download .xlsx
                  </a>
                </div>
              </div>
              {trainingTableActive && (
                <div className="overflow-x-auto border border-gray-200 rounded-lg">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="bg-gray-50">
                        <th className="px-2 py-1 text-xs text-gray-800 w-8">#</th>
                        {trainingTableCols.map((_, ci) => (
                          <th key={ci} className={`px-1 py-1 text-xs font-bold text-white text-center ${
                            ci < templateLabels ? 'bg-blue-500' : ci < templateLabels + templateInputs ? 'bg-green-500' : 'bg-red-500'
                          }`}>
                            {ci < templateLabels ? 'Label' : ci < templateLabels + templateInputs ? 'Input' : 'Output'}
                          </th>
                        ))}
                        <th className="w-8"></th>
                      </tr>
                      <tr className="bg-gray-50 border-b border-gray-300">
                        <td className="px-2 py-1 text-xs text-gray-800">Name</td>
                        {trainingTableCols.map((col, ci) => (
                          <td key={ci} className="px-1 py-1">
                            <input type="text" value={col} onChange={e => updateColumnName(trainingTableCols, setTrainingTableCols, ci, e.target.value)}
                              className={`w-full px-2 py-1 text-xs font-semibold text-center border border-gray-200 rounded ${
                                ci < templateLabels ? 'bg-blue-50' : ci < templateLabels + templateInputs ? 'bg-green-50' : 'bg-red-50'
                              }`} />
                          </td>
                        ))}
                        <td></td>
                      </tr>
                    </thead>
                    <tbody>
                      {trainingTableRows.map((row, ri) => (
                        <tr key={ri} className="border-b border-gray-100 hover:bg-gray-50">
                          <td className="px-2 py-1 text-xs text-gray-800 text-center">{ri + 1}</td>
                          {row.map((cell, ci) => (
                            <td key={ci} className="px-1 py-1">
                              <input type="text" value={cell} onChange={e => updateTableCell(trainingTableRows, setTrainingTableRows, ri, ci, e.target.value)}
                                onPaste={e => handleTablePaste(e, trainingTableRows, setTrainingTableRows, ri, ci, trainingTableCols.length)}
                                className="w-full px-2 py-1 text-xs border border-gray-200 rounded text-center focus:ring-1 focus:ring-blue-400 focus:border-blue-400" />
                            </td>
                          ))}
                          <td className="px-1 py-1">
                            <button onClick={() => removeRow(trainingTableRows, setTrainingTableRows, ri)} className="text-red-400 hover:text-red-600 text-xs" title="Remove row">&times;</button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <div className="p-2 bg-gray-50 border-t border-gray-200">
                    <button onClick={() => addRow(trainingTableRows, setTrainingTableRows, trainingTableCols.length)}
                      className="text-xs text-blue-600 hover:text-blue-800 font-medium">+ Add Row</button>
                    <span className="text-xs text-gray-800 ml-4">{trainingTableRows.length} rows &times; {trainingTableCols.length} columns</span>
                  </div>
                </div>
              )}
              {!trainingTableActive && (
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center text-gray-800">
                  <p>Click "Create Table" to start entering training data directly in the app</p>
                  <p className="text-xs mt-1">Or download the .xlsx template and upload it in the next step</p>
                </div>
              )}
            </div>

            {/* Prediction Data Table */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h2 className="text-lg font-bold text-gray-900">Samples Prediction Table</h2>
                  <p className="text-gray-800 text-sm">Enter prediction samples (no output column). Can also be done in Step 6.</p>
                </div>
                <div className="flex gap-2">
                  <button onClick={initPredictionTable}
                    className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 transition-colors">
                    {predictionTableActive ? 'Reset Table' : 'Create Table'}
                  </button>
                  <a href={`${API_URL}/api/template/prediction?n_labels=${templateLabels}&n_inputs=${templateInputs}`}
                    className="px-4 py-2 bg-gray-100 text-gray-800 text-sm font-medium rounded-lg hover:bg-gray-200 transition-colors">
                    Download .xlsx
                  </a>
                </div>
              </div>
              {predictionTableActive && (
                <div className="overflow-x-auto border border-gray-200 rounded-lg">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="bg-gray-50">
                        <th className="px-2 py-1 text-xs text-gray-800 w-8">#</th>
                        {predictionTableCols.map((_, ci) => (
                          <th key={ci} className={`px-1 py-1 text-xs font-bold text-white text-center ${
                            ci < templateLabels ? 'bg-blue-500' : 'bg-green-500'
                          }`}>
                            {ci < templateLabels ? 'Label' : 'Input'}
                          </th>
                        ))}
                        <th className="w-8"></th>
                      </tr>
                      <tr className="bg-gray-50 border-b border-gray-300">
                        <td className="px-2 py-1 text-xs text-gray-800">Name</td>
                        {predictionTableCols.map((col, ci) => (
                          <td key={ci} className="px-1 py-1">
                            <input type="text" value={col} onChange={e => updateColumnName(predictionTableCols, setPredictionTableCols, ci, e.target.value)}
                              className={`w-full px-2 py-1 text-xs font-semibold text-center border border-gray-200 rounded ${
                                ci < templateLabels ? 'bg-blue-50' : 'bg-green-50'
                              }`} />
                          </td>
                        ))}
                        <td></td>
                      </tr>
                    </thead>
                    <tbody>
                      {predictionTableRows.map((row, ri) => (
                        <tr key={ri} className="border-b border-gray-100 hover:bg-gray-50">
                          <td className="px-2 py-1 text-xs text-gray-800 text-center">{ri + 1}</td>
                          {row.map((cell, ci) => (
                            <td key={ci} className="px-1 py-1">
                              <input type="text" value={cell} onChange={e => updateTableCell(predictionTableRows, setPredictionTableRows, ri, ci, e.target.value)}
                                onPaste={e => handleTablePaste(e, predictionTableRows, setPredictionTableRows, ri, ci, predictionTableCols.length)}
                                className="w-full px-2 py-1 text-xs border border-gray-200 rounded text-center focus:ring-1 focus:ring-blue-400 focus:border-blue-400" />
                            </td>
                          ))}
                          <td className="px-1 py-1">
                            <button onClick={() => removeRow(predictionTableRows, setPredictionTableRows, ri)} className="text-red-400 hover:text-red-600 text-xs" title="Remove row">&times;</button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <div className="p-2 bg-gray-50 border-t border-gray-200">
                    <button onClick={() => addRow(predictionTableRows, setPredictionTableRows, predictionTableCols.length)}
                      className="text-xs text-blue-600 hover:text-blue-800 font-medium">+ Add Row</button>
                    <span className="text-xs text-gray-800 ml-4">{predictionTableRows.length} rows &times; {predictionTableCols.length} columns</span>
                  </div>
                </div>
              )}
              {!predictionTableActive && (
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center text-gray-800">
                  <p>Click "Create Table" to start entering prediction samples</p>
                  <p className="text-xs mt-1">Or download the .xlsx template and upload a file later</p>
                </div>
              )}
            </div>

            <div className="text-center">
              <button onClick={() => setStep('upload')}
                className="px-8 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition-colors">
                Continue to Training Data Upload
              </button>
            </div>
          </div>
        )}

        {step === 'upload' && (
          <div className="max-w-2xl mx-auto space-y-6">
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Training Data Upload</h2>
              <p className="text-gray-800 mb-4">Choose your data source for training.</p>
              <div className="flex gap-2 mb-6">
                <button onClick={() => setDataSource('table')}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${dataSource === 'table' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-800 hover:bg-gray-200'}`}>
                  Use Table Data {trainingTableActive ? `(${trainingTableRows.length} rows)` : '(not configured)'}
                </button>
                <button onClick={() => setDataSource('file')}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${dataSource === 'file' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-800 hover:bg-gray-200'}`}>
                  Upload File
                </button>
              </div>
              {dataSource === 'table' && trainingTableActive && (
                <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
                  <p className="text-green-800 font-medium text-sm">Training data from table: {trainingTableRows.length} rows &times; {trainingTableCols.length} columns</p>
                  <p className="text-green-600 text-xs mt-1">Data entered in Step 1 will be used. You can go back to edit it.</p>
                </div>
              )}
              {dataSource === 'table' && !trainingTableActive && (
                <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                  <p className="text-yellow-800 font-medium text-sm">No table data configured</p>
                  <p className="text-yellow-600 text-xs mt-1">Go back to Step 1 to create and fill the training data table, or switch to "Upload File" mode.</p>
                </div>
              )}
              {dataSource === 'file' && (
                <>
                <p className="text-gray-800 mb-4">Upload a TSV, CSV, or Excel file with your training data.</p>
                <div className="mb-6">
                  <label className="block text-sm font-medium text-gray-800 mb-2">Training Data <span className="text-red-500">*</span></label>
                  <div className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${trainingFile ? 'border-green-400 bg-green-50' : 'border-gray-300 hover:border-blue-400'}`}>
                    <input type="file" accept=".tsv,.csv,.xlsx,.xls,.txt" onChange={e => setTrainingFile(e.target.files?.[0] || null)} className="hidden" id="training-file" />
                    <label htmlFor="training-file" className="cursor-pointer">
                      {trainingFile ? (
                        <div>
                          <p className="font-medium text-green-700">{trainingFile.name}</p>
                          <p className="text-sm text-green-600">{(trainingFile.size / 1024).toFixed(1)} KB</p>
                        </div>
                      ) : (
                        <div>
                          <svg className="w-10 h-10 mx-auto text-gray-800 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                          </svg>
                          <p className="text-gray-800">Click to select or drag & drop</p>
                          <p className="text-sm text-gray-800 mt-1">TSV, CSV, or Excel files</p>
                        </div>
                      )}
                    </label>
                  </div>
                </div>
                </>
              )}
              {uploadError && <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">{uploadError}</div>}
              <button onClick={handleUpload} disabled={(dataSource === 'file' && !trainingFile) || (dataSource === 'table' && !trainingTableActive) || uploading}
                className="w-full py-3 px-4 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors">
                {uploading ? 'Uploading...' : 'Continue to Training Configuration'}
              </button>
            </div>
          </div>
        )}

        {step === 'config' && (
          <div className="max-w-4xl mx-auto">
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Training Configuration</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <div className="col-span-full"><h3 className="text-lg font-semibold text-gray-800 mb-3 pb-2 border-b">Data Split</h3></div>
                {[['Train %', 'train_percent'], ['Validation %', 'val_percent'], ['Test %', 'test_percent']].map(([label, key]) => (
                  <div key={key}>
                    <label className="block text-sm font-medium text-gray-800 mb-1">{label}</label>
                    <input type="number" value={config[key as keyof typeof config] as number} onChange={e => updateConfig(key, Number(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" />
                  </div>
                ))}

                <div className="col-span-full"><h3 className="text-lg font-semibold text-gray-800 mb-3 pb-2 border-b">Data Structure</h3></div>
                <div>
                  <label className="block text-sm font-medium text-gray-800 mb-1">Label Columns</label>
                  <input type="number" value={config.n_labels} onChange={e => updateConfig('n_labels', Number(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" />
                  <p className="text-xs text-gray-800 mt-1">Non-input label columns at start</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-800 mb-1">Input Columns</label>
                  <input type="number" value={config.n_inputs} onChange={e => updateConfig('n_inputs', Number(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-800 mb-1">Separator</label>
                  <select value={config.sep} onChange={e => updateConfig('sep', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                    <option value="\\t">Tab (TSV)</option>
                    <option value=",">Comma (CSV)</option>
                    <option value=";">Semicolon</option>
                  </select>
                </div>

                <div className="col-span-full"><h3 className="text-lg font-semibold text-gray-800 mb-3 pb-2 border-b">Model Training</h3></div>
                {[['Tuner Trials', 'tuner_trials'], ['K-Folds', 'k_folds'], ['Random Seed', 'random_seed'],
                  ['Tuner Epochs', 'tuner_epochs'], ['CV Epochs', 'cv_epochs'], ['Final Epochs', 'final_epochs']].map(([label, key]) => (
                  <div key={key}>
                    <label className="block text-sm font-medium text-gray-800 mb-1">{label}</label>
                    <input type="number" value={config[key as keyof typeof config] as number} onChange={e => updateConfig(key, Number(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" />
                  </div>
                ))}

                <div className="col-span-full"><h3 className="text-lg font-semibold text-gray-800 mb-3 pb-2 border-b">Advanced</h3></div>
                <div className="flex items-center gap-3">
                  <input type="checkbox" id="retrain" checked={config.do_optional_retrain}
                    onChange={e => updateConfig('do_optional_retrain', e.target.checked)} className="w-4 h-4 text-blue-600 rounded" />
                  <label htmlFor="retrain" className="text-sm font-medium text-gray-800">Optional retrain on Train+Val</label>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-800 mb-1">PI Calibration</label>
                  <select value={config.pi_calibration} onChange={e => updateConfig('pi_calibration', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                    <option value="val">Validation</option>
                    <option value="oof">OOF (Cross-validation)</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-800 mb-1">PI Alpha</label>
                  <input type="number" step="0.01" value={config.pi_alpha} onChange={e => updateConfig('pi_alpha', Number(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" />
                  <p className="text-xs text-gray-800 mt-1">0.05 = 95% prediction interval</p>
                </div>
              </div>
              <div className="flex gap-3 mt-8">
                <button onClick={() => setStep('upload')} className="px-6 py-3 border border-gray-300 text-gray-800 font-medium rounded-lg hover:bg-gray-50 transition-colors">Back to Upload</button>
                <button onClick={startTraining} className="flex-1 py-3 px-4 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition-colors">Start Training</button>
              </div>
            </div>
          </div>
        )}

        {step === 'running' && (
          <div className="max-w-4xl mx-auto space-y-6">
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-gray-900">Training in Progress</h2>
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                  jobStatus?.status === 'failed' ? 'bg-red-100 text-red-700' :
                  jobStatus?.status === 'completed' ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700'
                }`}>{jobStatus?.status || 'starting'}</span>
              </div>
              <div className="mb-2">
                <div className="flex justify-between text-sm text-gray-800 mb-1">
                  <span>{jobStatus?.stage || 'Initializing...'}</span>
                  <span>{jobStatus?.progress || 0}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div className="bg-blue-600 h-3 rounded-full transition-all duration-500" style={{ width: `${jobStatus?.progress || 0}%` }} />
                </div>
              </div>
              {jobStatus?.error && (
                <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-red-700 text-sm font-mono whitespace-pre-wrap">{jobStatus.error}</p>
                </div>
              )}
            </div>
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-3">Training Logs</h3>
              <div className="bg-gray-900 rounded-lg p-4 h-64 overflow-y-auto font-mono text-sm text-green-400">
                {logs.map((log, i) => <div key={i} className="py-0.5">{log}</div>)}
                <div ref={logsEndRef} />
                {logs.length === 0 && <div className="text-gray-800 animate-pulse">Waiting for logs...</div>}
              </div>
            </div>
          </div>
        )}

        {step === 'results' && jobStatus && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {(['train', 'val', 'test'] as const).map(split => {
                const m = jobStatus.metrics[split] as Record<string, unknown> | undefined
                if (!m) return null
                return (
                  <div key={split} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                    <h3 className="text-lg font-semibold text-gray-800 mb-3 capitalize">{split} Metrics</h3>
                    <div className="space-y-2">
                      {['R2', 'R2_adj', 'RMSE', 'MAE', 'MSE'].map(k => {
                        const val = m[k]; if (val === undefined) return null
                        return (<div key={k} className="flex justify-between"><span className="text-sm text-gray-800">{k}</span><span className="text-sm font-mono font-medium text-gray-900">{fmtVal(val)}</span></div>)
                      })}
                      {split === 'train' && m['Predicted_R2_Q2'] !== undefined && (
                        <div className="flex justify-between pt-2 border-t"><span className="text-sm text-gray-800">Q2</span><span className="text-sm font-mono font-medium text-gray-900">{fmtVal(m['Predicted_R2_Q2'])}</span></div>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {jobStatus.metrics.cv ? (
                <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                  <h3 className="text-lg font-semibold text-gray-800 mb-3">Cross-Validation</h3>
                  <div className="space-y-2">
                    {Object.entries(jobStatus.metrics.cv as Record<string, unknown>).map(([k, v]) => (
                      <div key={k} className="flex justify-between"><span className="text-sm text-gray-800">{k}</span><span className="text-sm font-mono font-medium text-gray-900">{fmtVal(v)}</span></div>
                    ))}
                  </div>
                </div>
              ) : null}
              {jobStatus.metrics.best_hyperparameters ? (
                <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                  <h3 className="text-lg font-semibold text-gray-800 mb-3">Best Hyperparameters</h3>
                  <div className="space-y-2">
                    {Object.entries(jobStatus.metrics.best_hyperparameters as Record<string, unknown>).map(([k, v]) => (
                      <div key={k} className="flex justify-between"><span className="text-sm text-gray-800">{k}</span><span className="text-sm font-mono font-medium text-gray-900">{String(v)}</span></div>
                    ))}
                    {jobStatus.metrics.best_epoch !== undefined && (
                      <div className="flex justify-between pt-2 border-t"><span className="text-sm text-gray-800">Best Epoch</span><span className="text-sm font-mono font-medium text-gray-900">{String(jobStatus.metrics.best_epoch)}</span></div>
                    )}
                  </div>
                </div>
              ) : null}
            </div>

            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-800">Plots</h3>
                <div className="flex gap-2">
                  {(['diagnostics', 'residuals', 'surfaces'] as const).map(cat => (
                    <button key={cat} onClick={() => setSelectedPlotCategory(cat)}
                      className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
                        selectedPlotCategory === cat ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-800 hover:bg-gray-200'
                      }`}>
                      {cat.charAt(0).toUpperCase() + cat.slice(1)} ({catPlots[cat].length})
                    </button>
                  ))}
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {catPlots[selectedPlotCategory as keyof typeof catPlots].map(plotName => (
                  <div key={plotName} className="border border-gray-200 rounded-lg overflow-hidden">
                    <div className="p-2 bg-gray-50 border-b"><p className="text-xs font-medium text-gray-800 truncate">{plotName.replace(/_/g, ' ')}</p></div>
                    {plots[plotName] ? (
                      <img src={`data:image/png;base64,${plots[plotName]}`} alt={plotName} className="w-full h-auto" />
                    ) : (
                      <div className="h-48 flex items-center justify-center text-gray-800 text-sm">Loading...</div>
                    )}
                  </div>
                ))}
                {catPlots[selectedPlotCategory as keyof typeof catPlots].length === 0 && (
                  <p className="text-gray-800 text-sm col-span-full text-center py-8">No plots in this category</p>
                )}
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Download Results</h3>
              <div className="flex flex-wrap gap-4">
                {jobStatus.has_results_zip && (
                  <a href={`${API_URL}/api/job/${jobId}/download/results`}
                    className="inline-flex items-center gap-2 px-6 py-3 bg-green-600 text-white font-medium rounded-lg hover:bg-green-700 transition-colors">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    Training Results (ZIP)
                  </a>
                )}
                {jobStatus.has_predictions_zip && (
                  <a href={`${API_URL}/api/job/${jobId}/download/predictions`}
                    className="inline-flex items-center gap-2 px-6 py-3 bg-purple-600 text-white font-medium rounded-lg hover:bg-purple-700 transition-colors">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    Predictions (ZIP)
                  </a>
                )}
              </div>
            </div>

            <div className="flex justify-center gap-4">
              <button onClick={() => setStep('prediction')}
                className="px-8 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition-colors">
                Continue to Samples Prediction
              </button>
              <button onClick={() => { setStep('templates'); setTrainingFile(null); setTrainingFileId(''); setNewDataFile(null); setNewDataFileId(''); setJobId(''); setJobStatus(null); setLogs([]); setLogSince(0); setPlots({}) }}
                className="px-8 py-3 bg-gray-800 text-white font-medium rounded-lg hover:bg-gray-900 transition-colors">
                Start New Run
              </button>
            </div>
          </div>
        )}

        {step === 'prediction' && (
          <div className="max-w-4xl mx-auto space-y-6">
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Samples Prediction</h2>
              <p className="text-gray-800 mb-4">Provide new samples to generate predictions using the trained model.</p>

              {/* Prediction table */}
              {predictionTableActive && (
                <div className="mb-6">
                  <h3 className="text-sm font-semibold text-gray-800 mb-2">Prediction Data from Table ({predictionTableRows.length} rows)</h3>
                  <div className="overflow-x-auto border border-gray-200 rounded-lg max-h-64 overflow-y-auto">
                    <table className="w-full text-sm">
                      <thead className="sticky top-0">
                        <tr className="bg-gray-50">
                          <th className="px-2 py-1 text-xs text-gray-800 w-8">#</th>
                          {predictionTableCols.map((_, ci) => (
                            <th key={ci} className={`px-1 py-1 text-xs font-bold text-white text-center ${
                              ci < templateLabels ? 'bg-blue-500' : 'bg-green-500'
                            }`}>
                              {ci < templateLabels ? 'Label' : 'Input'}
                            </th>
                          ))}
                          <th className="w-8"></th>
                        </tr>
                        <tr className="bg-gray-50 border-b border-gray-300">
                          <td className="px-2 py-1 text-xs text-gray-800">Name</td>
                          {predictionTableCols.map((col, ci) => (
                            <td key={ci} className="px-1 py-1">
                              <input type="text" value={col} onChange={e => updateColumnName(predictionTableCols, setPredictionTableCols, ci, e.target.value)}
                                className={`w-full px-2 py-1 text-xs font-semibold text-center border border-gray-200 rounded ${
                                  ci < templateLabels ? 'bg-blue-50' : 'bg-green-50'
                                }`} />
                            </td>
                          ))}
                          <td></td>
                        </tr>
                      </thead>
                      <tbody>
                        {predictionTableRows.map((row, ri) => (
                          <tr key={ri} className="border-b border-gray-100 hover:bg-gray-50">
                            <td className="px-2 py-1 text-xs text-gray-800 text-center">{ri + 1}</td>
                            {row.map((cell, ci) => (
                              <td key={ci} className="px-1 py-1">
                                <input type="text" value={cell} onChange={e => updateTableCell(predictionTableRows, setPredictionTableRows, ri, ci, e.target.value)}
                                  onPaste={e => handleTablePaste(e, predictionTableRows, setPredictionTableRows, ri, ci, predictionTableCols.length)}
                                  className="w-full px-2 py-1 text-xs border border-gray-200 rounded text-center focus:ring-1 focus:ring-blue-400 focus:border-blue-400" />
                              </td>
                            ))}
                            <td className="px-1 py-1">
                              <button onClick={() => removeRow(predictionTableRows, setPredictionTableRows, ri)} className="text-red-400 hover:text-red-600 text-xs" title="Remove row">&times;</button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <div className="p-2 bg-gray-50 border-t border-gray-200 sticky bottom-0">
                      <button onClick={() => addRow(predictionTableRows, setPredictionTableRows, predictionTableCols.length)}
                        className="text-xs text-blue-600 hover:text-blue-800 font-medium">+ Add Row</button>
                      <span className="text-xs text-gray-800 ml-4">{predictionTableRows.length} rows &times; {predictionTableCols.length} columns</span>
                    </div>
                  </div>
                </div>
              )}
              {!predictionTableActive && (
                <div className="mb-6">
                  <button onClick={initPredictionTable}
                    className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 transition-colors">
                    Create Prediction Table
                  </button>
                  <span className="text-xs text-gray-800 ml-3">Or upload a file below</span>
                </div>
              )}

              {/* File upload fallback */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-800 mb-2">Or Upload Prediction File</label>
                <div className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors ${newDataFile ? 'border-green-400 bg-green-50' : 'border-gray-300 hover:border-blue-400'}`}>
                  <input type="file" accept=".tsv,.csv,.xlsx,.xls,.txt" onChange={e => setNewDataFile(e.target.files?.[0] || null)} className="hidden" id="prediction-file" />
                  <label htmlFor="prediction-file" className="cursor-pointer">
                    {newDataFile ? (
                      <div>
                        <p className="font-medium text-green-700">{newDataFile.name}</p>
                        <p className="text-sm text-green-600">{(newDataFile.size / 1024).toFixed(1)} KB</p>
                      </div>
                    ) : (
                      <p className="text-gray-800 text-sm">Click to upload prediction samples file (.xlsx, .csv, .tsv)</p>
                    )}
                  </label>
                </div>
              </div>

              <p className="text-sm text-gray-800 mb-4">Input columns must match the training data. Use the table above or upload a file prepared with the Samples Prediction Template from Step 1.</p>
              <div className="flex gap-3">
                <button onClick={() => setStep('results')} className="px-6 py-3 border border-gray-300 text-gray-800 font-medium rounded-lg hover:bg-gray-50 transition-colors">Back to Results</button>
                <button onClick={() => { setStep('templates'); setTrainingFile(null); setTrainingFileId(''); setNewDataFile(null); setNewDataFileId(''); setJobId(''); setJobStatus(null); setLogs([]); setLogSince(0); setPlots({}) }}
                  className="flex-1 py-3 px-4 bg-gray-800 text-white font-medium rounded-lg hover:bg-gray-900 transition-colors">
                  Start New Run
                </button>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
