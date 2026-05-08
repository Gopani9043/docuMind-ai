import { useState } from 'react'
import axios from 'axios'

const API = 'http://localhost:8000'

export function useUpload() {
  const [progress, setProgress]   = useState(0)
  const [status, setStatus]       = useState('idle')
  const [error, setError]         = useState(null)
  const [docId, setDocId]         = useState(null)

  async function upload(file) {
    // Validate
    const allowed = ['application/pdf','image/png','image/jpeg','image/tiff']
    if (!allowed.includes(file.type)) {
      setError('Only PDF, PNG, JPG and TIFF files are supported.')
      return
    }
    if (file.size > 20 * 1024 * 1024) {
      setError('File too large. Maximum size is 20MB.')
      return
    }

    try {
      setStatus('uploading')
      setError(null)
      setProgress(0)

      // POST file to backend
      const form = new FormData()
      form.append('file', file)
      const { data } = await axios.post(`${API}/upload`, form, {
        onUploadProgress: e => setProgress(Math.round(e.loaded / e.total * 100))
      })

      // Poll for result
      setStatus('processing')
      setDocId(data.doc_id)

      const interval = setInterval(async () => {
        const { data: result } = await axios.get(`${API}/results/${data.doc_id}`)
        if (result.status === 'done' || result.status === 'error') {
          clearInterval(interval)
          setStatus(result.status)
        }
      }, 2000)

    } catch (e) {
      setError(e.response?.data?.detail || 'Upload failed.')
      setStatus('error')
    }
  }

  function reset() {
    setStatus('idle')
    setProgress(0)
    setError(null)
    setDocId(null)
  }

  return { upload, progress, status, error, docId, reset }
}