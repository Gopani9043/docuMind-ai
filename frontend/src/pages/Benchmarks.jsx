import { useState, useEffect } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function Benchmarks() {
  const [docs, setDocs] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    axios.get(`${API}/documents`)
      .then(r => { setDocs(r.data); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  if (loading) return <div className="text-center py-20 text-gray-400">Loading...</div>

  const total = docs.length
  const done  = docs.filter(d => d.status === 'done').length
  const err   = docs.filter(d => d.status === 'error').length
  const rate  = total ? Math.round(done / total * 100) : 0

  const byType = docs.reduce((acc, doc) => {
    const ext = doc.filename.split('.').pop().toUpperCase()
    acc[ext] = (acc[ext] || 0) + 1
    return acc
  }, {})

  return (
    <div className="max-w-3xl mx-auto">

      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900 mb-1">Benchmarks</h1>
          <p className="text-gray-500 text-sm">Real metrics from your processed documents</p>
        </div>
        <a href={`${API}/export/csv`} className="px-4 py-2 bg-green-600 text-white text-sm font-medium rounded-lg hover:bg-green-700 transition-colors">Export CSV</a>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        {[
          { label: 'Total processed', value: total, color: 'text-gray-900' },
          { label: 'Successful',      value: done,  color: 'text-green-600' },
          { label: 'Failed',          value: err,   color: 'text-red-500' },
          { label: 'Success rate',    value: `${rate}%`, color: 'text-green-600' },
        ].map(({ label, value, color }) => (
          <div key={label} className="bg-white border border-gray-200 rounded-xl p-5">
            <div className={`text-3xl font-semibold ${color} mb-1`}>{value}</div>
            <div className="text-xs text-gray-400">{label}</div>
          </div>
        ))}
      </div>

      {Object.keys(byType).length > 0 && (
        <div className="bg-white border border-gray-200 rounded-xl overflow-hidden mb-6">
          <div className="px-5 py-3.5 border-b border-gray-100">
            <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">By file type</h2>
          </div>
          <div className="divide-y divide-gray-50">
            {Object.entries(byType).map(([type, count]) => (
              <div key={type} className="flex items-center gap-4 px-5 py-3">
                <span className="text-xs font-medium text-gray-500 w-12">{type}</span>
                <div className="flex-1 bg-gray-100 rounded-full h-2">
                  <div className="bg-green-500 h-2 rounded-full" style={{ width: `${Math.round(count / total * 100)}%` }} />
                </div>
                <span className="text-xs text-gray-400 w-8 text-right">{count}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
        <div className="px-5 py-3.5 border-b border-gray-100">
          <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">All documents</h2>
        </div>
        {docs.length === 0 ? (
          <p className="px-5 py-8 text-center text-gray-400 text-sm">No documents yet — upload one to see it here.</p>
        ) : (
          <div className="divide-y divide-gray-50">
            {docs.map(doc => (
              <div key={doc.doc_id} className="flex items-center gap-4 px-5 py-3">
                <span className="text-sm text-gray-700 flex-1 truncate">{doc.filename}</span>
                <span className="text-xs text-gray-400">{new Date(doc.created_at).toLocaleDateString()}</span>
                <span className={`text-xs px-2.5 py-1 rounded-full font-medium ${doc.status === 'done' ? 'bg-green-50 text-green-700' : doc.status === 'error' ? 'bg-red-50 text-red-600' : 'bg-amber-50 text-amber-600'}`}>{doc.status}</span>
              </div>
            ))}
          </div>
        )}
      </div>

    </div>
  )
}