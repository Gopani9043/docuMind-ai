import { useState, useEffect, useRef } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function QA() {
  const [docs, setDocs]           = useState([])
  const [selectedDoc, setSelected] = useState(null)
  const [messages, setMessages]   = useState([])
  const [question, setQuestion]   = useState('')
  const [loading, setLoading]     = useState(false)
  const bottomRef                 = useRef(null)

  // Load documents on mount
  useEffect(() => {
    axios.get(`${API}/documents`)
      .then(r => setDocs(r.data.filter(d => d.status === 'done')))
      .catch(() => {})
  }, [])

  // Auto scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Clear chat when document changes
  function handleDocChange(e) {
    const doc = docs.find(d => d.doc_id === e.target.value) || null
    setSelected(doc)
    setMessages([{
      role: 'ai',
      text: doc
        ? `I have loaded ${doc.filename}. Ask me anything about this document!`
        : 'No document selected. Ask me any general question about document processing.'
    }])
  }

  async function sendQuestion() {
    if (!question.trim() || loading) return

    const q = question.trim()
    setQuestion('')
    setMessages(prev => [...prev, { role: 'user', text: q }])
    setLoading(true)

    try {
      const { data } = await axios.post(`${API}/qa`, {
        question: q,
        doc_id: selectedDoc?.doc_id || null
      })

      setMessages(prev => [...prev, {
        role: 'ai',
        text: data.answer,
        sources: data.sources || [],
        mode: data.mode
      }])
    } catch (e) {
      setMessages(prev => [...prev, {
        role: 'ai',
        text: 'Sorry, something went wrong. Please try again.',
        error: true
      }])
    } finally {
      setLoading(false)
    }
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendQuestion()
    }
  }

  const suggestions = selectedDoc
    ? ['What is the termination clause?', 'When does this document expire?', 'What are the payment terms?', 'Who are the parties involved?']
    : ['What is OCR?', 'How does LLM extraction work?', 'What document types are supported?']

  return (
    <div className="max-w-5xl mx-auto">

      {/* HEADER */}
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 mb-1">Document Q&A</h1>
        <p className="text-gray-500 text-sm">Ask questions about any document or ask general questions</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

        {/* LEFT PANEL */}
        <div className="flex flex-col gap-4">

          {/* Document selector */}
          <div>
            <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
              Select document (optional)
            </label>
            <select
              onChange={handleDocChange}
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm text-gray-700 bg-white focus:outline-none focus:border-green-400"
            >
              <option value="">No document — general Q&A</option>
              {docs.map(doc => (
                <option key={doc.doc_id} value={doc.doc_id}>
                  {doc.filename}
                </option>
              ))}
            </select>
          </div>

          {/* Selected document info */}
          {selectedDoc && (
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
              <p className="text-sm font-medium text-gray-800 truncate">{selectedDoc.filename}</p>
              <p className="text-xs text-gray-400 mt-1">
                {new Date(selectedDoc.created_at).toLocaleDateString()}
              </p>
              <span className="inline-block mt-2 text-xs px-2 py-1 rounded-full bg-green-50 text-green-700 font-medium">
                {selectedDoc.document_type || 'document'}
              </span>
            </div>
          )}

          {/* Suggested questions */}
          <div>
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
              Suggested questions
            </p>
            <div className="flex flex-col gap-2">
              {suggestions.map(s => (
                <button
                  key={s}
                  onClick={() => setQuestion(s)}
                  className="text-left text-xs text-gray-500 border border-gray-200 rounded-lg px-3 py-2 hover:border-green-300 hover:text-gray-700 transition-colors bg-white"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>

        </div>

        {/* RIGHT PANEL — Chat */}
        <div className="md:col-span-2 flex flex-col border border-gray-200 rounded-xl overflow-hidden" style={{height: '520px'}}>

          {/* Chat header */}
          <div className="px-4 py-3 border-b border-gray-100 flex items-center justify-between bg-white">
            <span className="text-sm font-medium text-gray-700">
              {selectedDoc ? `Asking about: ${selectedDoc.filename}` : 'General Q&A'}
            </span>
            <span className="text-xs text-gray-400">Powered by RAG + Llama 3.3</span>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4 bg-gray-50">
            {messages.length === 0 && (
              <div className="text-center py-12 text-gray-400 text-sm">
                Select a document or ask a general question to get started
              </div>
            )}

            {messages.map((msg, i) => (
              <div key={i} className={`flex gap-2 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>

                {msg.role === 'ai' && (
                  <div className="w-7 h-7 rounded-full bg-green-100 flex items-center justify-center text-green-700 text-xs font-medium flex-shrink-0 mt-1">
                    AI
                  </div>
                )}

                <div className={`max-w-lg ${msg.role === 'user' ? 'items-end' : 'items-start'} flex flex-col gap-1`}>
                  <div className={`px-3 py-2 rounded-xl text-sm leading-relaxed ${
                    msg.role === 'user'
                      ? 'bg-green-600 text-white'
                      : msg.error
                      ? 'bg-red-50 text-red-600 border border-red-200'
                      : 'bg-white border border-gray-200 text-gray-800'
                  }`}>
                    {msg.text}
                  </div>

                  {/* Sources */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="text-xs text-gray-400 bg-white border border-gray-100 rounded-lg px-3 py-2 border-l-2 border-l-green-400 max-w-lg">
                      <span className="font-medium text-gray-500">📄 Source: </span>
                      {msg.sources[0].substring(0, 120)}...
                    </div>
                  )}
                </div>

                {msg.role === 'user' && (
                  <div className="w-7 h-7 rounded-full bg-green-600 flex items-center justify-center text-white text-xs font-medium flex-shrink-0 mt-1">
                    You
                  </div>
                )}

              </div>
            ))}

            {loading && (
              <div className="flex gap-2 justify-start">
                <div className="w-7 h-7 rounded-full bg-green-100 flex items-center justify-center text-green-700 text-xs font-medium flex-shrink-0">
                  AI
                </div>
                <div className="px-3 py-2 rounded-xl bg-white border border-gray-200 text-gray-400 text-sm">
                  Thinking...
                </div>
              </div>
            )}

            <div ref={bottomRef} />
          </div>

          {/* Input */}
          <div className="px-4 py-3 border-t border-gray-100 bg-white flex gap-2">
            <input
              value={question}
              onChange={e => setQuestion(e.target.value)}
              onKeyDown={handleKey}
              placeholder="Ask a question... (Enter to send)"
              className="flex-1 border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-green-400"
            />
            <button
              onClick={sendQuestion}
              disabled={loading || !question.trim()}
              className="px-4 py-2 bg-green-600 text-white text-sm font-medium rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Send
            </button>
          </div>

        </div>
      </div>
    </div>
  )
}