import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import axios from 'axios'

const API = 'http://localhost:8000'

export default function Results() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showJson, setShowJson] = useState(false)

  useEffect(() => {
    if (!id) { setLoading(false); return }
    axios.get(`${API}/results/${id}`)
      .then(r => { setResult(r.data); setLoading(false) })
      .catch(() => { setError('Document not found'); setLoading(false) })
  }, [id])

  // No doc selected
  if (!id) return (
    <div className="max-w-2xl mx-auto text-center py-20">
      <div className="text-4xl mb-4">📋</div>
      <p className="text-gray-500">Upload a document to see results here.</p>
      <button
        onClick={() => navigate('/')}
        className="mt-4 text-sm text-green-600 underline"
      >
        Go to Upload
      </button>
    </div>
  )

  if (loading) return (
    <div className="max-w-2xl mx-auto text-center py-20">
      <div className="text-2xl animate-spin inline-block mb-4">⚙️</div>
      <p className="text-gray-400">Loading results...</p>
    </div>
  )

  if (error) return (
    <div className="max-w-2xl mx-auto text-center py-20">
      <p className="text-red-500">{error}</p>
      <button onClick={() => navigate('/')} className="mt-4 text-sm text-green-600 underline">
        Go to Upload
      </button>
    </div>
  )

  const fields = result.extracted_data || {}
  const confidence = result.confidence || {}

  return (
    <div className="max-w-3xl mx-auto">

      {/* HEADER */}
      <div className="flex items-center gap-4 mb-8">
        <button
          onClick={() => navigate('/')}
          className="text-sm text-gray-400 hover:text-gray-600"
        >
          ← Back
        </button>
        <div className="flex-1">
          <h1 className="text-2xl font-semibold text-gray-900">
            Extraction Results
          </h1>
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

    </div>
  )
}