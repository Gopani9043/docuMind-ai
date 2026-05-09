import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useUpload } from '../hooks/useUpload'
import axios from 'axios'

const API = 'http://localhost:8000'

export default function Upload() {
  const navigate = useNavigate()
  const { upload, progress, status, error, docId, reset } = useUpload()
  const [dragOver, setDragOver] = useState(false)
  const [recentDocs, setRecentDocs] = useState([])

  // Load recent documents on mount
  useState(() => {
    axios.get(`${API}/documents`)
      .then(r => setRecentDocs(r.data.slice(0, 5)))
      .catch(() => {})
  }, [])

  // Navigate to results when done
  if (status === 'done' && docId) {
    navigate(`/results/${docId}`)
  }

  const handleDrop = useCallback(e => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) upload(file)
  }, [upload])

  const handleChange = e => {
    const file = e.target.files[0]
    if (file) upload(file)
  }

  return (
    <div className="max-w-2xl mx-auto">

      {/* HEADER */}
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-gray-900 mb-1">
          Upload a document
        </h1>
        <p className="text-gray-500 text-sm">
          Supports PDF, PNG, JPG and TIFF — invoices, contracts, receipts, reports
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
        <input
          id="file-input"
          type="file"
          className="hidden"
          accept=".pdf,.png,.jpg,.jpeg,.tiff"
          onChange={handleChange}
        />

      {/* BATCH UPLOAD */}
      <div className="mt-6">
        <label className="block text-sm font-medium text-gray-500 mb-2">
            Batch upload multiple documents
        </label>
        <input
            type="file"
            multiple
            accept=".pdf,.png,.jpg,.jpeg,.tiff"
            onChange={async (e) => {
            const files = Array.from(e.target.files)
            if (!files.length) return
            const form = new FormData()
            files.forEach(f => form.append('files', f))
            try {
                const { data } = await axios.post(`${API}/batch-upload`, form)
                alert(`Processed ${data.total} documents successfully!`)
                window.location.reload()
            } catch(err) {
                alert('Batch upload failed')
            }
            }}
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-green-50 file:text-green-700 hover:file:bg-green-100 cursor-pointer"
        />
        </div>
        
        {/* Idle state */}
        {status === 'idle' && (
          <>
            <div className="text-4xl mb-3">📄</div>
            <p className="text-gray-700 font-medium mb-1">
              Drop your document here
            </p>
            <p className="text-gray-400 text-sm">or click to browse</p>
            <div className="flex gap-2 justify-center mt-4 flex-wrap">
              {['Invoice','Contract','Receipt','Report'].map(t => (
                <span key={t} className="text-xs bg-gray-100 text-gray-500 px-3 py-1 rounded-full">
                  {t}
                </span>
              ))}
            </div>
          </>
        )}

        {/* Uploading state */}
        {status === 'uploading' && (
          <div className="space-y-3">
            <div className="text-2xl">⬆️</div>
            <p className="text-gray-700 font-medium">Uploading...</p>
            <div className="w-full bg-gray-100 rounded-full h-2 max-w-xs mx-auto">
              <div
                className="bg-green-500 h-2 rounded-full transition-all"
                style={{ width: `${progress}%` }}
              />
            </div>
            <p className="text-gray-400 text-sm">{progress}%</p>
          </div>
        )}

        {/* Processing state */}
        {status === 'processing' && (
          <div className="space-y-3">
            <div className="text-2xl animate-spin inline-block">⚙️</div>
            <p className="text-gray-700 font-medium">Processing document...</p>
            <p className="text-gray-400 text-sm">
              Running OCR and LLM extraction
            </p>
          </div>
        )}

        {/* Error state */}
        {status === 'error' && (
          <div className="space-y-3">
            <div className="text-2xl">❌</div>
            <p className="text-red-600 font-medium">{error || 'Processing failed'}</p>
            <button
              onClick={e => { e.stopPropagation(); reset() }}
              className="text-sm text-green-600 underline"
            >
              Try again
            </button>
          </div>
        )}
      </div>

      {/* ERROR BANNER */}
      {error && status === 'idle' && (
        <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm">
          {error}
        </div>
      )}

      {/* RECENT DOCUMENTS */}
      {recentDocs.length > 0 && (
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
                  <p className="text-sm font-medium text-gray-800 truncate">
                    {doc.filename}
                  </p>
                  <p className="text-xs text-gray-400">
                    {new Date(doc.created_at).toLocaleDateString()}
                  </p>
                </div>
                <span className={`text-xs px-2.5 py-1 rounded-full font-medium ${
                  doc.status === 'done'
                    ? 'bg-green-50 text-green-700'
                    : doc.status === 'error'
                    ? 'bg-red-50 text-red-600'
                    : 'bg-amber-50 text-amber-600'
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