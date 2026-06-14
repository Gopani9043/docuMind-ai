import { useState, useEffect, useRef } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const AIAvatar = () => (
  <div style={{
    width:'30px',height:'30px',borderRadius:'50%',
    background:'#EAF4EE',display:'flex',alignItems:'center',
    justifyContent:'center',flexShrink:0
  }}>
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0F6E56" strokeWidth="2" strokeLinecap="round">
      <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2z"/>
      <circle cx="9" cy="14" r="1" fill="#0F6E56"/>
      <circle cx="15" cy="14" r="1" fill="#0F6E56"/>
    </svg>
  </div>
)

const UserAvatar = () => (
  <div style={{
    width:'30px',height:'30px',borderRadius:'50%',
    background:'#1D9E75',display:'flex',alignItems:'center',
    justifyContent:'center',flexShrink:0
  }}>
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round">
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
      <circle cx="12" cy="7" r="4"/>
    </svg>
  </div>
)

// ── localStorage helpers ──
const loadMessages = (key) => {
  try {
    const saved = localStorage.getItem(key)
    return saved ? JSON.parse(saved) : []
  } catch { return [] }
}

const saveMessages = (key, messages) => {
  try {
    // Only save last 50 messages to avoid localStorage limit
    localStorage.setItem(key, JSON.stringify(messages.slice(-50)))
  } catch {}
}

const SMART_MESSAGES_KEY  = 'docmind_smart_messages'
const DOC_MESSAGES_KEY    = 'docmind_doc_messages'

export default function QA() {
  const [docs, setDocs]         = useState([])
  const [selectedDoc, setSelected] = useState(null)
  const [question, setQuestion] = useState('')
  const [loading, setLoading]   = useState(false)
  const [mode, setMode]         = useState(() => {
    return localStorage.getItem('docmind_mode') || 'document'
  })
  const bottomRef = useRef(null)

  // ── Persistent session ID ──
  const [sessionId] = useState(() => {
    const existing = localStorage.getItem('docmind_session_id')
    if (existing) return existing
    const newId = `session_${Date.now()}`
    localStorage.setItem('docmind_session_id', newId)
    return newId
  })

  // ── Separate message history per mode ──
  const [smartMessages, setSmartMessages] = useState(() =>
    loadMessages(SMART_MESSAGES_KEY)
  )
  const [docMessages, setDocMessages] = useState(() =>
    loadMessages(DOC_MESSAGES_KEY)
  )

  // Current messages based on mode
  const messages = mode === 'smart' ? smartMessages : docMessages
  const setMessages = (updater) => {
    if (mode === 'smart') {
      setSmartMessages(prev => {
        const next = typeof updater === 'function' ? updater(prev) : updater
        saveMessages(SMART_MESSAGES_KEY, next)
        return next
      })
    } else {
      setDocMessages(prev => {
        const next = typeof updater === 'function' ? updater(prev) : updater
        saveMessages(DOC_MESSAGES_KEY, next)
        return next
      })
    }
  }

  useEffect(() => {
    axios.get(`${API}/documents`)
      .then(r => setDocs(r.data.filter(d => d.status === 'done')))
      .catch(() => {})
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Save mode to localStorage when it changes
  useEffect(() => {
    localStorage.setItem('docmind_mode', mode)
  }, [mode])

  function handleDocChange(e) {
    const doc = docs.find(d => d.doc_id === e.target.value) || null
    setSelected(doc)
    setDocMessages([{
      role: 'ai',
      text: doc
        ? `I have loaded ${doc.filename}. Ask me anything about this document!`
        : 'No document selected. Ask me any general question.'
    }])
    saveMessages(DOC_MESSAGES_KEY, [{
      role: 'ai',
      text: doc
        ? `I have loaded ${doc.filename}. Ask me anything about this document!`
        : 'No document selected. Ask me any general question.'
    }])
  }

  function handleModeChange(newMode) {
    setMode(newMode)
    setSelected(null)
    // Do NOT clear messages — restore from localStorage automatically
  }

  async function sendQuestion() {
    if (!question.trim() || loading) return
    const q = question.trim()
    setQuestion('')
    setMessages(prev => [...prev, { role: 'user', text: q }])
    setLoading(true)

    try {
      let data

      if (mode === 'smart') {
        const res = await axios.post(`${API}/smart-chat`, {
          question: q,
          doc_id: sessionId
        })
        data = res.data

        if (data.rewritten) {
          setMessages(prev => [...prev, {
            role: 'system',
            text: `🔄 Understood as: "${data.rewritten}"`
          }])
        }

        if (data.decomposed) {
          setMessages(prev => [...prev, {
            role: 'system',
            text: `🧩 Complex question — analyzed in multiple parts`
          }])
        }

      } else {
        if (selectedDoc) {
          try { await axios.get(`${API}/qa/index/${selectedDoc.doc_id}`) } catch(e) {}
        }
        const res = await axios.post(`${API}/qa`, {
          question: q,
          doc_id: selectedDoc?.doc_id || null
        })
        data = res.data
      }

      setMessages(prev => [...prev, {
        role: 'ai',
        text: data.answer,
        mode: data.mode,
        count: data.count,
        intent: data.intent
      }])

    } catch(e) {
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
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuestion() }
  }

  // ── Clear chat for smart mode ──
  async function handleClearChat() {
    try {
      await axios.delete(`${API}/smart-chat/session/${sessionId}`)
    } catch(e) {}
    localStorage.removeItem('docmind_session_id')
    localStorage.removeItem(SMART_MESSAGES_KEY)
    const newId = `session_${Date.now()}`
    localStorage.setItem('docmind_session_id', newId)
    setSmartMessages([{
      role: 'ai',
      text: 'Conversation cleared! What would you like to know?'
    }])
    saveMessages(SMART_MESSAGES_KEY, [{
      role: 'ai',
      text: 'Conversation cleared! What would you like to know?'
    }])
  }

  const documentSuggestions = selectedDoc
    ? ['What is the total amount?', 'When does this document expire?', 'What are the payment terms?', 'Who are the parties involved?']
    : ['What is OCR?', 'How does LLM extraction work?', 'What document types are supported?']

  const smartSuggestions = [
    'Show me all invoices',
    'Which vendor did I pay the most?',
    'Show documents uploaded today',
    'List all contracts',
    'Total amount of all invoices',
  ]

  const suggestions = mode === 'smart' ? smartSuggestions : documentSuggestions

  return (
    <div className="max-w-5xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-gray-900 mb-1">Document Q&A</h1>
        <p className="text-gray-500 text-sm">Ask questions about a specific document or query your entire document library</p>
      </div>

      {/* MODE SWITCHER */}
      <div className="flex gap-2 mb-6 items-center flex-wrap">
        <button
          onClick={() => handleModeChange('document')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            mode === 'document'
              ? 'bg-green-600 text-white'
              : 'bg-white border border-gray-200 text-gray-600 hover:border-green-300'
          }`}
        >
          📄 Document Q&A
        </button>
        <button
          onClick={() => handleModeChange('smart')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            mode === 'smart'
              ? 'bg-green-600 text-white'
              : 'bg-white border border-gray-200 text-gray-600 hover:border-green-300'
          }`}
        >
          🧠 Smart Chat
        </button>

        {mode === 'smart' && (
          <button
            onClick={handleClearChat}
            className="text-xs text-gray-400 hover:text-red-500 transition-colors"
          >
            Clear chat
          </button>
        )}

        {mode === 'smart' && (
          <span className="ml-2 text-xs text-gray-400 self-center">
            Query your entire document library with natural language
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

        {/* LEFT PANEL */}
        <div className="flex flex-col gap-4">

          {mode === 'document' && (
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
                  <option key={doc.doc_id} value={doc.doc_id}>{doc.filename}</option>
                ))}
              </select>
            </div>
          )}

          {mode === 'smart' && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-3">
              <p className="text-xs font-semibold text-green-700 mb-1">🧠 Smart Chat</p>
              <p className="text-xs text-green-600 leading-relaxed">
                Query all {docs.length} documents at once. Ask about amounts, vendors, dates, or any field.
              </p>
            </div>
          )}

          {selectedDoc && mode === 'document' && (
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
              <p className="text-sm font-medium text-gray-800 truncate">{selectedDoc.filename}</p>
              <p className="text-xs text-gray-400 mt-1">{new Date(selectedDoc.created_at).toLocaleDateString()}</p>
              <span className="inline-block mt-2 text-xs px-2 py-1 rounded-full bg-green-50 text-green-700 font-medium">
                {selectedDoc.document_type || 'document'}
              </span>
            </div>
          )}

          <div>
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
              {mode === 'smart' ? 'Example queries' : 'Suggested questions'}
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

        {/* CHAT PANEL */}
        <div className="md:col-span-2 flex flex-col border border-gray-200 rounded-xl overflow-hidden" style={{height:'520px'}}>

          <div className="px-4 py-3 border-b border-gray-100 flex items-center gap-3 bg-white">
            <div className="w-8 h-8 rounded-lg bg-green-50 flex items-center justify-center flex-shrink-0">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0F6E56" strokeWidth="2" strokeLinecap="round">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
              </svg>
            </div>
            <div>
              <div className="text-sm font-medium text-gray-700">
                {mode === 'smart' ? 'Smart Chat — All Documents' : selectedDoc ? `Asking about: ${selectedDoc.filename}` : 'General Q&A'}
              </div>
              <div className="text-xs text-gray-400">
                {mode === 'smart' ? 'Natural language → SQL → Answer' : 'Powered by RAG + Llama 3.3'}
              </div>
            </div>
            <span className="ml-auto text-xs px-2 py-1 rounded-full bg-green-50 text-green-700 font-medium">
              {mode === 'smart' ? '🧠 Smart' : '📄 RAG'}
            </span>
          </div>

          <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4 bg-gray-50">
            {messages.length === 0 && (
              <div className="text-center py-12 text-gray-400 text-sm">
                {mode === 'smart'
                  ? 'Ask anything about your document library'
                  : 'Select a document or ask a general question'}
              </div>
            )}

            {messages.map((msg, i) => (
              msg.role === 'system' ? (
                <div key={i} className="text-center">
                  <span className="text-xs text-gray-400 bg-gray-100 px-3 py-1 rounded-full">{msg.text}</span>
                </div>
              ) : (
                <div key={i} className={`flex gap-2 items-end ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                  {msg.role === 'ai' ? <AIAvatar /> : <UserAvatar />}
                  <div className="flex flex-col gap-1" style={{maxWidth:'72%'}}>
                    <div className={`px-3 py-2 rounded-xl text-sm leading-relaxed whitespace-pre-wrap ${
                      msg.role === 'user'
                        ? 'bg-green-600 text-white rounded-br-sm'
                        : msg.error
                        ? 'bg-red-50 text-red-600 border border-red-200'
                        : 'bg-white border border-gray-200 text-gray-800 rounded-bl-sm'
                    }`}>
                      {msg.text}
                    </div>
                    {msg.count !== undefined && msg.count > 0 && (
                      <span className="text-xs text-gray-400">{msg.count} result{msg.count !== 1 ? 's' : ''}</span>
                    )}
                  </div>
                </div>
              )
            ))}

            {loading && (
              <div className="flex gap-2 items-end">
                <AIAvatar />
                <div className="px-3 py-2 rounded-xl bg-white border border-gray-200 text-gray-400 text-sm rounded-bl-sm">
                  {mode === 'smart' ? 'Querying database...' : 'Thinking...'}
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="px-4 py-3 border-t border-gray-100 bg-white flex gap-2">
            <input
              value={question}
              onChange={e => setQuestion(e.target.value)}
              onKeyDown={handleKey}
              placeholder={mode === 'smart' ? 'e.g. Show all invoices above EUR 1000...' : 'Ask a question... (Enter to send)'}
              className="flex-1 border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-green-400"
            />
            <button
              onClick={sendQuestion}
              disabled={loading || !question.trim()}
              className="px-4 py-2 bg-green-600 text-white text-sm font-medium rounded-lg hover:bg-green-700 disabled:opacity-50 transition-colors"
            >
              Send
            </button>
          </div>

        </div>
      </div>
    </div>
  )
}