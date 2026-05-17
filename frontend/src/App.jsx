import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import Upload from './pages/Upload'
import Results from './pages/Results'
import Benchmarks from './pages/Benchmarks.jsx'
import QA from './pages/QA'

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50">

        {/* NAVBAR */}
        <nav className="bg-white border-b border-gray-200 sticky top-0 z-50">
          <div className="max-w-5xl mx-auto px-6 flex items-center gap-8 h-14">
            <span className="font-semibold text-gray-900 text-lg">DocParse</span>
            <div className="flex gap-1">
              {[
                { to: '/',            label: 'Upload'     },
                { to: '/results',     label: 'Results'    },
                { to: '/benchmarks',  label: 'Benchmarks' },
                { to: '/qa',          label: 'Document Q&A' },
              ].map(({ to, label }) => (
                <NavLink
                  key={to}
                  to={to}
                  end
                  className={({ isActive }) =>
                    `px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
                      isActive
                        ? 'bg-green-50 text-green-700'
                        : 'text-gray-500 hover:text-gray-900'
                    }`
                  }
                >
                  {label}
                </NavLink>
              ))}
            </div>
          </div>
        </nav>

        {/* PAGES */}
        <main className="max-w-5xl mx-auto px-6 py-10">
          <Routes>
            <Route path="/"           element={<Upload />} />
            <Route path="/results"    element={<Results />} />
            <Route path="/results/:id" element={<Results />} />
            <Route path="/benchmarks" element={<Benchmarks />} />
            <Route path="/qa"         element={<QA />} />
          </Routes>
        </main>

      </div>
    </BrowserRouter>
  )
}