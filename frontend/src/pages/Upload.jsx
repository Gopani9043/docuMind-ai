import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function Upload() {
  const navigate = useNavigate()
  const [dragOver, setDragOver]     = useState(false)
  const [files, setFiles]           = useState([])
  const [processing, setProcessing] = useState(false)
  const [results, setResults]       = useState([])
  const [recentDocs, setRecentDocs] = useState([])

  useState(() => {
    axios.get(`${API}/documents`)
      .then(r => setRecentDocs(r.data.slice(0, 5)))
      .catch(() => {})
  }, [])

  const handleDrop = useCallback(e => {
    e.preventDefault()
    setDragOver(false)
    const dropped = Array.from(e.dataTransfer.files)
    // ── Single file chooser: replace, don't append ──
    setFiles(dropped)
  }, [])

  const handleChange = e => {
    const selected = Array.from(e.target.files)
    // ── Single file chooser: replace, don't append ──
    setFiles(selected)
    e.target.value = ''
  }

  const removeFile = index => {
    setFiles(prev => prev.filter((_, i) => i !== index))
  }

  const processAll = async () => {
    if (!files.length || processing) return
    setProcessing(true)
    setResults([])

    const newResults = files.map(f => ({
      filename: f.name,
      status: 'queued',
      doc_id: null,
      error: null
    }))
    setResults(newResults)

    for (let i = 0; i < files.length; i++) {
      const file = files[i]

      const allowed = ['application/pdf', 'image/png', 'image/jpeg', 'image/tiff']
      if (!allowed.includes(file.type)) {
        setResults(prev => prev.map((r, idx) =>
          idx === i ? { ...r, status: 'error', error: 'Unsupported file type' } : r
        ))
        continue
      }

      setResults(prev => prev.map((r, idx) =>
        idx === i ? { ...r, status: 'uploading' } : r
      ))

      try {
        const form = new FormData()
        form.append('file', file)
        const { data } = await axios.post(`${API}/upload`, form)

        setResults(prev => prev.map((r, idx) =>
          idx === i ? { ...r, status: 'processing', doc_id: data.doc_id } : r
        ))

        // ── FIXED: Poll with retry — don't stop on 404 ──
        await new Promise(resolve => {
          let attempts = 0
          const maxAttempts = 150 // 5 minutes max

          const poll = setInterval(async () => {
            attempts++
            if (attempts > maxAttempts) {
              clearInterval(poll)
              setResults(prev => prev.map((r, idx) =>
                idx === i ? { ...r, status: 'error', error: 'Processing timed out' } : r
              ))
              resolve()
              return
            }

            try {
              const { data: result } = await axios.get(`${API}/results/${data.doc_id}`)
              if (result.status === 'done' || result.status === 'error') {
                clearInterval(poll)
                setResults(prev => prev.map((r, idx) =>
                  idx === i ? {
                    ...r,
                    status: result.status,
                    document_type: result.document_type
                  } : r
                ))
                resolve()
              }
              // if still processing — keep polling, don't stop
            } catch(e) {
              // ── FIXED: 404 means not ready yet, keep polling ──
              if (e.response?.status === 404) return
              // only stop on real errors
              clearInterval(poll)
              setResults(prev => prev.map((r, idx) =>
                idx === i ? {
                  ...r,
                  status: 'error',
                  error: 'Could not check status'
                } : r
              ))
              resolve()
            }
          }, 2000)
        })

      } catch(e) {
        setResults(prev => prev.map((r, idx) =>
          idx === i ? {
            ...r,
            status: 'error',
            error: e.response?.data?.detail || 'Upload failed'
          } : r
        ))
      }
    }

    setProcessing(false)
  }

  const clearAll = () => {
    setFiles([])
    setResults([])
  }

  const statusColor = status => {
    switch(status) {
      case 'done':       return 'bg-green-50 text-green-700'
      case 'error':      return 'bg-red-50 text-red-600'
      case 'processing': return 'bg-amber-50 text-amber-600'
      case 'uploading':  return 'bg-blue-50 text-blue-600'
      default:           return 'bg-gray-100 text-gray-500'
    }
  }

  const statusIcon = status => {
    switch(status) {
      case 'done':       return '✅'
      case 'error':      return '❌'
      case 'processing': return '⚙️'
      case 'uploading':  return '⬆️'
      default:           return '⏳'
    }
  }

  return (
    <div className="max-w-2xl mx-auto">

      {/* HEADER */}
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-gray-900 mb-1">Upload Documents</h1>
        <p className="text-gray-500 text-sm">
          Select one or more documents — PDF, PNG, JPG or TIFF
        </p>
      </div>

      {/* DROP ZONE */}
      <div
        onDrop={handleDrop}
        onDragOver={e => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onClick={() => document.getElementById('file-input').click()}
        className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors ${
          dragOver
            ? 'border-green-400 bg-green-50'
            : 'border-gray-200 bg-white hover:border-green-300 hover:bg-gray-50'
        }`}
      >
        {/* ── Single file input, multiple allowed ── */}
        <input
          id="file-input"
          type="file"
          className="hidden"
          accept=".pdf,.png,.jpg,.jpeg,.tiff"
          multiple
          onChange={handleChange}
        />
        <div className="text-4xl mb-3">📄</div>
        <p className="text-gray-700 font-medium mb-1">
          Drop documents here or click to browse
        </p>
        <p className="text-gray-400 text-sm">Select one or multiple files at once</p>
        <div className="flex gap-2 justify-center mt-4 flex-wrap">
          {['Invoice', 'Contract', 'Receipt', 'Report'].map(t => (
            <span key={t} className="text-xs bg-gray-100 text-gray-500 px-3 py-1 rounded-full">
              {t}
            </span>
          ))}
        </div>
      </div>

      {/* SELECTED FILES */}
      {files.length > 0 && results.length === 0 && (
        <div className="mt-6">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">
              {files.length} file{files.length > 1 ? 's' : ''} selected
            </h2>
            <button
              onClick={clearAll}
              className="text-xs text-gray-400 hover:text-red-500 transition-colors"
            >
              Clear all
            </button>
          </div>

          <div className="bg-white border border-gray-200 rounded-xl overflow-hidden mb-4">
            {files.map((file, i) => (
              <div
                key={i}
                className={`flex items-center gap-3 px-4 py-3 ${
                  i < files.length - 1 ? 'border-b border-gray-100' : ''
                }`}
              >
                <span className="text-lg">📄</span>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-800 truncate">{file.name}</p>
                  <p className="text-xs text-gray-400">{(file.size / 1024).toFixed(0)} KB</p>
                </div>
                <button
                  onClick={() => removeFile(i)}
                  className="text-gray-300 hover:text-red-400 transition-colors text-lg"
                >
                  ×
                </button>
              </div>
            ))}
          </div>

          {/* ── PROCESS BUTTON ── */}
          <button
            onClick={processAll}
            disabled={processing}
            className="w-full py-3 bg-green-600 text-white font-semibold rounded-xl hover:bg-green-700 disabled:opacity-50 transition-colors text-sm"
          >
            {processing
              ? `Processing...`
              : `Process ${files.length} document${files.length > 1 ? 's' : ''}`
            }
          </button>
        </div>
      )}

      {/* PROCESSING RESULTS */}
      {results.length > 0 && (
        <div className="mt-6">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">
              Processing results
            </h2>
            {!processing && (
              <button
                onClick={clearAll}
                className="text-xs text-green-600 hover:text-green-700 font-medium"
              >
                Upload more
              </button>
            )}
          </div>

          <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
            {results.map((result, i) => (
              <div
                key={i}
                onClick={() => result.doc_id && result.status === 'done' && navigate(`/results/${result.doc_id}`)}
                className={`flex items-center gap-3 px-4 py-3 ${
                  i < results.length - 1 ? 'border-b border-gray-100' : ''
                } ${result.status === 'done' ? 'cursor-pointer hover:bg-gray-50' : ''}`}
              >
                <span className="text-lg">{statusIcon(result.status)}</span>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-800 truncate">{result.filename}</p>
                  {result.document_type && (
                    <p className="text-xs text-gray-400 capitalize">{result.document_type}</p>
                  )}
                  {result.error && (
                    <p className="text-xs text-red-500">{result.error}</p>
                  )}
                </div>
                <span className={`text-xs px-2.5 py-1 rounded-full font-medium ${statusColor(result.status)}`}>
                  {result.status}
                </span>
              </div>
            ))}
          </div>

          {/* Summary */}
          {!processing && (
            <div className="mt-3 flex gap-4 text-xs text-gray-400 justify-center">
              <span>✅ {results.filter(r => r.status === 'done').length} done</span>
              <span>❌ {results.filter(r => r.status === 'error').length} failed</span>
              <span>⚙️ {results.filter(r => r.status === 'processing').length} processing</span>
            </div>
          )}
        </div>
      )}

      {/* RECENT DOCUMENTS */}
      {recentDocs.length > 0 && results.length === 0 && files.length === 0 && (
        <div className="mt-10">
          <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
            Recent uploads
          </h2>
          <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
            {recentDocs.map((doc, i) => (
              <div
                key={doc.doc_id}
                onClick={() => navigate(`/results/${doc.doc_id}`)}
                className={`flex items-center gap-4 px-5 py-3.5 cursor-pointer hover:bg-gray-50 transition-colors ${
                  i < recentDocs.length - 1 ? 'border-b border-gray-100' : ''
                }`}
              >
                <span className="text-xl">📄</span>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-800 truncate">{doc.filename}</p>
                  <p className="text-xs text-gray-400">
                    {new Date(doc.created_at).toLocaleDateString()}
                  </p>
                </div>
                <span className={`text-xs px-2.5 py-1 rounded-full font-medium ${
                  doc.status === 'done' ? 'bg-green-50 text-green-700' :
                  doc.status === 'error' ? 'bg-red-50 text-red-600' :
                  'bg-amber-50 text-amber-600'
                }`}>
                  {doc.status}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

    </div>
  )
}