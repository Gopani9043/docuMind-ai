import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function Results() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [result, setResult]     = useState(null)
  const [loading, setLoading]   = useState(true)
  const [error, setError]       = useState(null)
  const [showJson, setShowJson] = useState(false)
  const [allDocs, setAllDocs]   = useState([])

  // Load all done documents for sidebar
  useEffect(() => {
    axios.get(`${API}/documents`)
      .then(r => setAllDocs(r.data.filter(d => d.status === 'done')))
      .catch(() => {})
  }, [])

  // Load selected document result
  useEffect(() => {
    if (!id) { setLoading(false); return }
    setLoading(true)
    setError(null)
    setResult(null)
    axios.get(`${API}/results/${id}`)
      .then(r => { setResult(r.data); setLoading(false) })
      .catch(() => { setError('Document not found'); setLoading(false) })
  }, [id])

  const fields     = result?.extracted_data || {}
  const confidence = result?.confidence || {}

  return (
    <div className="max-w-6xl mx-auto">

      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 mb-1">Extraction Results</h1>
        <p className="text-gray-500 text-sm">Select a document from the list to view its extracted data</p>
      </div>

      <div className="flex gap-6">

        {/* ── SIDEBAR: All done documents ── */}
        <div className="w-64 flex-shrink-0">
          <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-100">
              <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Documents ({allDocs.length})
              </h2>
            </div>
            <div className="overflow-y-auto" style={{ maxHeight: '70vh' }}>
              {allDocs.length === 0 ? (
                <div className="px-4 py-6 text-center">
                  <p className="text-xs text-gray-400">No processed documents yet</p>
                  <button
                    onClick={() => navigate('/')}
                    className="mt-2 text-xs text-green-600 underline"
                  >
                    Upload documents
                  </button>
                </div>
              ) : (
                allDocs.map((doc, i) => (
                  <div
                    key={doc.doc_id}
                    onClick={() => navigate(`/results/${doc.doc_id}`)}
                    className={`flex items-start gap-3 px-4 py-3 cursor-pointer transition-colors ${
                      i < allDocs.length - 1 ? 'border-b border-gray-50' : ''
                    } ${
                      id === doc.doc_id
                        ? 'bg-green-50 border-l-2 border-l-green-500'
                        : 'hover:bg-gray-50'
                    }`}
                  >
                    <span className="text-base mt-0.5">📄</span>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-medium text-gray-800 truncate">
                        {doc.filename}
                      </p>
                      <p className="text-xs text-gray-400 mt-0.5">
                        {new Date(doc.created_at).toLocaleDateString()}
                      </p>
                      {doc.document_type && (
                        <span className="inline-block mt-1 text-xs px-1.5 py-0.5 rounded bg-green-50 text-green-700 capitalize">
                          {doc.document_type}
                        </span>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* ── MAIN CONTENT ── */}
        <div className="flex-1 min-w-0">

          {/* No document selected */}
          {!id && (
            <div className="text-center py-20 bg-white border border-gray-200 rounded-xl">
              <div className="text-4xl mb-4">📋</div>
              <p className="text-gray-500 text-sm">Select a document from the list to view results</p>
              <button
                onClick={() => navigate('/')}
                className="mt-4 text-sm text-green-600 underline"
              >
                Upload new document
              </button>
            </div>
          )}

          {/* Loading */}
          {id && loading && (
            <div className="text-center py-20 bg-white border border-gray-200 rounded-xl">
              <div className="text-2xl animate-spin inline-block mb-4">⚙️</div>
              <p className="text-gray-400 text-sm">Loading results...</p>
            </div>
          )}

          {/* Error */}
          {id && error && (
            <div className="text-center py-20 bg-white border border-gray-200 rounded-xl">
              <p className="text-red-500 text-sm">{error}</p>
              <button
                onClick={() => navigate('/')}
                className="mt-4 text-sm text-green-600 underline"
              >
                Go to Upload
              </button>
            </div>
          )}

          {/* Results */}
          {result && !loading && (
            <>
              {/* Document header */}
              <div className="flex items-center gap-3 mb-4 bg-white border border-gray-200 rounded-xl px-5 py-4">
                <span className="text-2xl">📄</span>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-gray-800 truncate">
                    {result.filename}
                  </p>
                  <p className="text-xs text-gray-400 mt-0.5">
                    {result.created_at
                      ? new Date(result.created_at).toLocaleString()
                      : ''}
                  </p>
                </div>
                <span className={`text-xs px-3 py-1 rounded-full font-medium ${
                  result.status === 'done'
                    ? 'bg-green-50 text-green-700'
                    : 'bg-red-50 text-red-600'
                }`}>
                  {result.document_type}
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">

                {/* EXTRACTED FIELDS */}
                <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
                  <div className="px-5 py-3.5 border-b border-gray-100">
                    <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">
                      Extracted Fields
                    </h2>
                  </div>
                  <div className="divide-y divide-gray-50">
                    {Object.keys(fields).length === 0 ? (
                      <p className="px-5 py-4 text-sm text-gray-400">No fields extracted</p>
                    ) : (
                      Object.entries(fields).map(([key, val]) => (
                        <div key={key} className="flex justify-between items-start px-5 py-3 gap-4">
                          <span className="text-xs text-gray-400 capitalize min-w-0 flex-shrink-0">
                            {key.replace(/_/g, ' ')}
                          </span>
                          <span className="text-sm font-medium text-gray-800 text-right break-all">
                            {Array.isArray(val)
                              ? `${val.length} items`
                              : val === null ? '—' : String(val)
                            }
                          </span>
                        </div>
                      ))
                    )}
                  </div>
                </div>

                {/* CONFIDENCE SCORES */}
                <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
                  <div className="px-5 py-3.5 border-b border-gray-100">
                    <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">
                      Confidence Scores
                    </h2>
                  </div>
                  <div className="divide-y divide-gray-50">
                    {Object.keys(confidence).length === 0 ? (
                      <p className="px-5 py-4 text-sm text-gray-400">No scores available</p>
                    ) : (
                      Object.entries(confidence).map(([key, val]) => (
                        <div key={key} className="px-5 py-3">
                          <div className="flex justify-between mb-1.5">
                            <span className="text-xs text-gray-400 capitalize">
                              {key.replace(/_/g, ' ')}
                            </span>
                            <span className="text-xs font-medium text-gray-700">
                              {Math.round(val * 100)}%
                            </span>
                          </div>
                          <div className="w-full bg-gray-100 rounded-full h-1.5">
                            <div
                              className={`h-1.5 rounded-full ${
                                val >= 0.9 ? 'bg-green-500' :
                                val >= 0.7 ? 'bg-amber-400' : 'bg-red-400'
                              }`}
                              style={{ width: `${val * 100}%` }}
                            />
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>

              {/* RAW JSON */}
              <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
                <button
                  onClick={() => setShowJson(v => !v)}
                  className="w-full flex items-center justify-between px-5 py-3.5 hover:bg-gray-50 transition-colors"
                >
                  <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">
                    Raw JSON Output
                  </h2>
                  <span className="text-xs text-gray-400">{showJson ? '▲ hide' : '▼ show'}</span>
                </button>
                {showJson && (
                  <div className="border-t border-gray-100">
                    <pre className="p-5 text-xs text-gray-700 bg-gray-50 overflow-x-auto leading-relaxed">
                      {JSON.stringify(result.extracted_data, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}