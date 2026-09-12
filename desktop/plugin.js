import {
  PALETTE_AREA,
  ROUTES_AREA,
  SIDEBAR_NAV_AREA,
  host,
} from '@hermes/plugin-sdk'
import { useCallback, useEffect, useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

let api = null

const fieldStyle = {
  width: '100%',
  boxSizing: 'border-box',
  border: '1px solid var(--ui-stroke-secondary)',
  borderRadius: '6px',
  padding: '7px 9px',
  background: 'transparent',
  color: 'var(--ui-text-primary)',
}

const buttonStyle = {
  border: '1px solid var(--ui-stroke-secondary)',
  borderRadius: '6px',
  padding: '7px 11px',
  background: 'transparent',
  color: 'var(--ui-text-primary)',
  cursor: 'pointer',
}


function Spark({ values }) {
  const nums = (values || []).map((value) => Number(value) || 0)
  if (nums.length < 2) return jsx('div', { style: { height: '36px' } })
  const max = Math.max(...nums, 1)
  const points = nums.map((value, index) => {
    const x = (index / (nums.length - 1)) * 120
    const y = 32 - (value / max) * 28
    return `${x},${y}`
  }).join(' ')
  return jsx('svg', {
    viewBox: '0 0 120 36',
    style: { width: '100%', height: '36px' },
    children: jsx('polyline', {
      fill: 'none',
      stroke: 'var(--ui-accent)',
      strokeWidth: '2',
      points,
    }),
  })
}

function ScoreBar({ label, value, max = 100 }) {
  const numeric = Number(value) || 0
  const ceiling = Number(max) || 100
  const pct = Math.max(0, Math.min(100, (numeric / ceiling) * 100))
  return jsxs('div', {
    className: 'flex flex-col gap-1',
    children: [
      jsxs('div', {
        className: 'flex justify-between text-xs',
        style: { color: 'var(--ui-text-secondary)' },
        children: [
          jsx('span', { children: label }),
          jsx('span', { children: Number.isFinite(numeric) ? String(value) : '—' }),
        ],
      }),
      jsx('div', {
        style: {
          height: '7px',
          borderRadius: '99px',
          background: 'var(--ui-stroke-secondary)',
          overflow: 'hidden',
        },
        children: jsx('div', {
          style: {
            width: `${pct}%`,
            height: '100%',
            borderRadius: '99px',
            background: 'var(--ui-accent)',
          },
        }),
      }),
    ],
  })
}

function Field({ label, children, help }) {
  return jsxs('label', {
    className: 'flex flex-col gap-1 text-xs',
    children: [
      jsx('span', { className: 'font-medium', children: label }),
      children,
      help ? jsx('span', {
        style: { color: 'var(--ui-text-tertiary)' },
        children: help,
      }) : null,
    ],
  })
}

function TurbofitPage() {
  const [status, setStatus] = useState(null)
  const [combinations, setCombinations] = useState([])
  const [recommendations, setRecommendations] = useState(null)
  const [multimodal, setMultimodal] = useState(null)
  const [multimodalSelections, setMultimodalSelections] = useState({})
  const [preference, setPreference] = useState('balanced')

  const [selectionMode, setSelectionMode] = useState('auto')
  const [mainModel, setMainModel] = useState('')
  const [auxModel, setAuxModel] = useState('')
  const [context, setContext] = useState('')
  const [primary, setPrimary] = useState(false)
  const [fallback, setFallback] = useState(false)
  const [baseUrl, setBaseUrl] = useState('http://127.0.0.1:8091/v1')
  const [fallbackChain, setFallbackChain] = useState([])
  const [usageHistory, setUsageHistory] = useState([])
  const [publishTailnet, setPublishTailnet] = useState(false)
  const [installSirvir, setInstallSirvir] = useState(false)
  const [installNative, setInstallNative] = useState(false)
  const [installFreeToken, setInstallFreeToken] = useState(false)
  const [installLemonade, setInstallLemonade] = useState(false)
  const [dashboardLocalPort, setDashboardLocalPort] = useState('9127')
  const [providerLocalPort, setProviderLocalPort] = useState('8091')
  const [dashboardHttpsPort, setDashboardHttpsPort] = useState('9444')
  const [providerHttpsPort, setProviderHttpsPort] = useState('9443')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [smoke, setSmoke] = useState(null)
  const [replacement, setReplacement] = useState(null)
  const [audition, setAudition] = useState(null)

  const refresh = useCallback(async () => {
    setBusy(true)
    setMessage('')
    try {
      const [nextStatus, nextCombinations, nextRecommendations, nextMultimodal, nextReplacement] = await Promise.all([
        api.rest('/status'),
        api.rest('/combinations'),
        api.rest(`/recommendations?preference=${encodeURIComponent(preference)}`),
        api.rest('/multimodal'),
        api.rest('/local-models').catch(() => ({ offer: false })),
      ])
      setStatus(nextStatus)
      const exact = (nextCombinations && nextCombinations.combinations) || []
      setCombinations(exact)
      setRecommendations(nextRecommendations)
      setMultimodal(nextMultimodal)
      setReplacement(nextReplacement && nextReplacement.offer ? nextReplacement : null)
      const selectedModalities = { ...(nextMultimodal.selected || {}) }
      ;['image', 'video', 'music', 'tts', 'stt'].forEach((modality) => {
        if (!selectedModalities[modality]) {
          const rows = (nextMultimodal.modalities && nextMultimodal.modalities[modality]) || []
          const choice = rows.find((item) => item.recommended_fit) || rows.find((item) => item.fit)
          if (choice) selectedModalities[modality] = choice.id
        }
      })
      setMultimodalSelections(selectedModalities)
      const provider = (nextStatus && nextStatus.provider) || {}
      setPrimary(Boolean(provider.primary))
      setFallback(Boolean(provider.fallback))
      setBaseUrl(provider.base_url || 'http://127.0.0.1:8091/v1')
      setFallbackChain(Array.isArray(provider.fallback_chain) ? provider.fallback_chain : [])
      const usage = nextStatus && nextStatus.usage
      if (usage) {
        setUsageHistory((prev) => [...prev, {
          tps: Number(usage.tps) || 0,
          vram: ((usage.gpus || []).reduce((sum, gpu) => sum + (Number(gpu.used_mb) || 0), 0)),
        }].slice(-24))
      }
      const requested = (nextStatus && nextStatus.selection && nextStatus.selection.requested) || 'auto'

      const exactId = requested.startsWith('manual-') ? requested.slice(7) : requested
      const selected = exact.find((item) => item.profile === exactId)
      if (selected) {
        setSelectionMode('manual')
        setMainModel(selected.main)
        setAuxModel(selected.aux)
        setContext(String(selected.context))
      } else {
        setSelectionMode('auto')
        const first = exact.find((item) => item.fit) || exact[0]
        if (first) {
          setMainModel(first.main)
          setAuxModel(first.aux)
          setContext(String(first.context))
        }
      }
    } catch (error) {
      setMessage(String(error && error.message || error))
    } finally {
      setBusy(false)
    }
  }, [preference])

  useEffect(() => { refresh() }, [refresh])

  async function save() {
    setBusy(true)
    setMessage('')
    try {
      const chain = Array.isArray(fallbackChain) ? fallbackChain : []
      const chosen = combinations.find((item) =>
        item.main === mainModel && item.aux === auxModel && String(item.context) === String(context)
      )
      if (selectionMode === 'manual' && !chosen) throw new Error('Choose an exact main, auxiliary, and context combination.')
      if (selectionMode === 'manual' && !chosen.fit) throw new Error(chosen.fit_reason || 'This combination does not fit this machine.')
      const result = await api.rest('/configure', {
        method: 'POST',
        body: {
          primary,
          fallback,
          fallback_chain: chain,
          multimodal: multimodalSelections,
          profile: selectionMode === 'auto' ? 'auto' : chosen.profile,
          base_url: baseUrl,
          publish_tailnet: publishTailnet,
          install_sirvir: installSirvir,
          install_native: installNative,
          install_freetoken: installFreeToken,
          install_lemonade: installLemonade,
          dashboard_local_port: Number(dashboardLocalPort),
          provider_local_port: Number(providerLocalPort),
          dashboard_https_port: Number(dashboardHttpsPort),
          provider_https_port: Number(providerHttpsPort),
        },
      })
      setMessage(result.restart_required ? 'Saved. Start a new Hermes session to use provider changes.' : 'Saved.')
      await refresh()
    } catch (error) {
      setMessage(String(error && error.message || error))
    } finally {
      setBusy(false)
    }
  }

  async function runShift(target) {
    setBusy(true)
    setMessage('')
    try {
      const result = await api.rest('/shift', { method: 'POST', body: { target } })
      setMessage(result.reason || result.error || (result.shifted ? `Shifted to ${result.profile}` : 'Shift complete'))
      await refresh()
    } catch (error) {
      setMessage(String(error && error.message || error))
    } finally {
      setBusy(false)
    }
  }

  async function runAudition() {
    setBusy(true)
    setMessage('')
    try {
      const pair = mainModel || 'maple-preview-tq2'
      const result = await api.rest(`/audition?main=${encodeURIComponent(pair)}&aux=${encodeURIComponent(auxModel || 'auto')}&context=${encodeURIComponent(context || '65536')}`)
      setAudition(result)
      setMessage(`Auditioned ${pair} across ${((result && result.engines) || []).length} engines.`)
    } catch (error) {
      setMessage(String(error && error.message || error))
    } finally {
      setBusy(false)
    }
  }

  async function runRetire(action) {
    if (!replacement) return
    setBusy(true)
    setMessage('')
    try {
      const result = await api.rest('/retire-model', {
        method: 'POST',
        body: { family: replacement.from_family, action },
      })
      setMessage(result.message || `${action} complete`)
      await refresh()
    } catch (error) {
      setMessage(String(error && error.message || error))
    } finally {
      setBusy(false)
    }
  }

  async function runUpdate() {
    setBusy(true)
    setMessage('')
    try {
      const result = await api.rest('/update', { method: 'POST', body: {} })
      setMessage(result.message || 'Updated Turbofit and Sirvir on this device.')
      await refresh()
    } catch (error) {
      setMessage(String(error && error.message || error))
    } finally {
      setBusy(false)
    }
  }

  async function runServe() {
    setBusy(true)
    setMessage('')
    try {
      const result = await api.rest('/serve', { method: 'POST', body: {} })
      setMessage(result.message || result.provider_base_url || 'Published on Tailscale Serve.')
      await refresh()
    } catch (error) {
      setMessage(String(error && error.message || error))
    } finally {
      setBusy(false)
    }
  }

  async function runSmoke() {
    setBusy(true)
    setMessage('Running local-runtime-smoke-v1 against 127.0.0.1:8091…')
    setSmoke(null)
    try {
      const result = await api.rest('/smoke', {
        method: 'POST',
        body: { timeout_seconds: 300 },
      })
      setSmoke(result)
      setMessage(result.message || (result.ok ? 'Local smoke passed. Nothing was promoted.' : 'Local smoke failed. Nothing was promoted.'))
    } catch (error) {
      setMessage(String(error && error.message || error))
    } finally {
      setBusy(false)
    }
  }

  const choices = ((recommendations && recommendations.recommendations) || {})[preference] || []
  const compatibleLanes = (recommendations && recommendations.compatible_lanes) || []
  const hardware = (recommendations && recommendations.hardware) || {}

  const mainModels = [...new Set(combinations.map((item) => item.main))]
  const mainRows = combinations.filter((item) => item.main === mainModel)
  const auxModels = [...new Set(mainRows.map((item) => item.aux))]
  const pairRows = mainRows.filter((item) => item.aux === auxModel)
  const contexts = [...new Set(pairRows.map((item) => item.context))].sort((a, b) => a - b)
  const selectedCombination = pairRows.find((item) => String(item.context) === String(context))

  return jsxs('div', {
    className: 'flex h-full flex-col gap-4 overflow-auto p-5 text-sm',
    children: [
      jsxs('div', { children: [
        jsx('h1', { className: 'text-xl font-semibold', children: 'Turbofit' }),
        jsx('p', {
          style: { color: 'var(--ui-text-tertiary)' },
          children: `Hardware: ${hardware.topology_key || 'scanning'} · usable memory ${hardware.total_usable_memory_mb || '—'} MiB`,
        }),
      ] }),
      jsxs('section', {
        className: 'grid grid-cols-1 gap-3 md:grid-cols-3',
        children: [
          jsxs('div', { style: { border: '1px solid var(--ui-stroke-secondary)', borderRadius: '6px', padding: '9px' }, children: [
            jsx('div', { className: 'font-medium', children: 'Speed' }),
            jsx(ScoreBar, { label: 'tok/s', value: (status && status.usage && status.usage.tps) || 0, max: 80 }),
            jsx(Spark, { values: usageHistory.map((item) => item.tps) }),
          ] }),
          jsxs('div', { style: { border: '1px solid var(--ui-stroke-secondary)', borderRadius: '6px', padding: '9px' }, children: [
            jsx('div', { className: 'font-medium', children: 'VRAM' }),
            jsx(ScoreBar, {
              label: 'used MiB',
              value: ((status && status.usage && status.usage.gpus) || []).reduce((sum, gpu) => sum + (Number(gpu.used_mb) || 0), 0),
              max: Math.max(1, ((status && status.usage && status.usage.gpus) || []).reduce((sum, gpu) => sum + (Number(gpu.total_mb) || 0), 0)),
            }),
            jsx(Spark, { values: usageHistory.map((item) => item.vram) }),
          ] }),
          jsxs('div', { style: { border: '1px solid var(--ui-stroke-secondary)', borderRadius: '6px', padding: '9px' }, children: [
            jsx('div', { className: 'font-medium', children: 'Host RAM' }),
            jsx(ScoreBar, {
              label: 'usable MiB',
              value: (status && status.usage && status.usage.host && status.usage.host.host_usable_memory_mb) || 0,
              max: Math.max(1, (status && status.usage && status.usage.host && status.usage.host.system_ram_mb) || 1),
            }),
          ] }),
        ],
      }),
      jsxs('section', { className: 'flex flex-col gap-2', children: [
        jsx('h2', { className: 'font-semibold', children: 'Auto scale down / up' }),
        jsx('p', { style: { color: 'var(--ui-text-tertiary)' }, children: 'Pressure drops context, then model, then Nous keyless. Healing walks back up.' }),
        jsx('div', { className: 'flex flex-wrap gap-2', children: ((status && status.scale_ladder) || []).map((step, index) => jsxs('div', {
          key: step.id,
          style: { border: '1px solid var(--ui-stroke-secondary)', borderRadius: '6px', padding: '8px' },
          children: [
            jsx('div', { className: 'font-medium', children: `${index + 1}. ${step.label}` }),
            jsx('div', { style: { color: 'var(--ui-text-tertiary)' }, children: `${step.aux_mode || ''} ${step.context ? `· ${step.context}` : ''}` }),
          ],
        })) }),
      ] }),
      message ? jsx('div', {
        style: { border: '1px solid var(--ui-stroke-secondary)', borderRadius: '6px', padding: '8px' },
        children: message,
      }) : null,
      jsxs('div', {
        className: 'flex flex-wrap gap-2',
        children: [
          jsx('button', { type: 'button', disabled: busy, style: buttonStyle, onClick: () => runShift('up'), children: 'Shift up' }),
          jsx('button', { type: 'button', disabled: busy, style: buttonStyle, onClick: () => runShift('down'), children: 'Shift down' }),
          jsx('button', { type: 'button', disabled: busy, style: buttonStyle, onClick: () => runShift(preference), children: `Shift ${preference}` }),
          jsx('button', { type: 'button', disabled: busy, style: buttonStyle, onClick: runUpdate, children: 'Update Turbofit + Sirvir' }),
          jsx('button', { type: 'button', disabled: busy, style: buttonStyle, onClick: runServe, children: 'Serve on Tailscale' }),
          jsx('button', { type: 'button', disabled: busy, style: buttonStyle, onClick: runSmoke, children: 'Smoke local runtime' }),
          jsx('button', { type: 'button', disabled: busy, style: buttonStyle, onClick: runAudition, children: 'Audition engines' }),
        ],
      }),
      replacement ? jsxs('div', {
        style: {
          border: '2px solid #d9aa50',
          background: 'rgba(217,170,80,0.12)',
          borderRadius: '8px',
          padding: '12px',
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
        },
        children: [
          jsx('strong', { children: 'New model recommended' }),
          jsx('span', { children: replacement.prompt || `Check recommends ${replacement.to_family} instead of ${replacement.from_family}.` }),
          jsx('span', { style: { color: 'var(--ui-text-secondary)' }, children: 'Archive keeps the old weights off the live path. Delete frees the disk. Keep both leaves them installed.' }),
          jsxs('div', {
            className: 'flex flex-wrap gap-2',
            children: [
              jsx('button', { type: 'button', disabled: busy, style: { ...buttonStyle, borderColor: '#d9aa50' }, onClick: () => setReplacement(null), children: 'Keep both' }),
              jsx('button', { type: 'button', disabled: busy, style: { ...buttonStyle, borderColor: '#d9aa50' }, onClick: () => runRetire('archive'), children: 'Archive old model' }),
              jsx('button', { type: 'button', disabled: busy, style: { ...buttonStyle, borderColor: '#d9aa50', color: '#d9aa50' }, onClick: () => runRetire('delete'), children: 'Delete old model' }),
            ],
          }),
        ],
      }) : null,
      jsxs('div', {
        className: 'grid grid-cols-1 gap-3 md:grid-cols-2',
        children: [
          jsx(Field, {
            label: 'Recommendation priority',
            children: jsxs('select', {
              value: preference,
              onChange: (event) => setPreference(event.target.value),
              style: fieldStyle,
              children: [
                jsx('option', { value: 'intelligence', children: 'Intelligence' }),
                jsx('option', { value: 'balanced', children: 'Balanced' }),
                jsx('option', { value: 'speed', children: 'Tokens per second' }),
              ],
            }),
          }),
          jsx(Field, {
            label: 'Runtime selection',
            help: 'Auto chooses an exact-hardware winner. Manual exposes physical-fit lanes and clearly marks those that still need an on-box benchmark.',
            children: jsxs('select', {
              value: selectionMode,
              onChange: (event) => setSelectionMode(event.target.value),
              style: fieldStyle,
              children: [
                jsx('option', { value: 'auto', children: 'Auto — recommended' }),
                jsx('option', { value: 'manual', children: `Manual — ${combinations.filter((item) => item.fit).length} compatible combinations` }),
              ],
            }),
          }),
          selectionMode === 'manual' ? jsx(Field, {
            label: 'Main model',
            children: jsxs('select', {
              value: mainModel,
              onChange: (event) => {
                const value = event.target.value
                const rows = combinations.filter((item) => item.main === value)
                const first = rows.find((item) => item.fit) || rows[0]
                setMainModel(value)
                setAuxModel(first ? first.aux : '')
                setContext(first ? String(first.context) : '')
              },
              style: fieldStyle,
              children: mainModels.map((model) => {
                const rows = combinations.filter((item) => item.main === model)
                const fits = rows.some((item) => item.fit)
                const label = rows[0] ? `${rows[0].main_name} · ${rows[0].main_quantization}` : model
                return jsx('option', { value: model, disabled: !fits, children: `${label}${fits ? '' : ' — unavailable'}` }, model)
              }),
            }),
          }) : null,
          selectionMode === 'manual' ? jsx(Field, {
            label: 'Auxiliary model',
            children: jsxs('select', {
              value: auxModel,
              onChange: (event) => {
                const value = event.target.value
                const rows = mainRows.filter((item) => item.aux === value)
                const first = rows.find((item) => item.fit) || rows[0]
                setAuxModel(value)
                setContext(first ? String(first.context) : '')
              },
              style: fieldStyle,
              children: auxModels.map((model) => {
                const rows = mainRows.filter((item) => item.aux === model)
                const fits = rows.some((item) => item.fit)
                const label = rows[0] ? `${rows[0].aux_name} · ${rows[0].aux_quantization}` : model
                return jsx('option', { value: model, disabled: !fits, children: `${label}${fits ? '' : ' — unavailable'}` }, model)
              }),
            }),
          }) : null,
          selectionMode === 'manual' ? jsx(Field, {
            label: 'Context length',
            children: jsxs('select', {
              value: context,
              onChange: (event) => setContext(event.target.value),
              style: fieldStyle,
              children: contexts.map((value) => {
                const row = pairRows.find((item) => item.context === value)
                return jsx('option', {
                  value: String(value),
                  disabled: !row.fit,
                  children: `${value.toLocaleString()} tokens${row.fit ? '' : ' — does not fit'}`,
                }, String(value))
              }),
            }),
          }) : null,
          selectionMode === 'manual' && selectedCombination ? jsx('div', {
            style: { border: '1px solid var(--ui-stroke-secondary)', borderRadius: '6px', padding: '8px' },
            children: jsxs('div', { className: 'flex flex-col gap-1 text-xs', children: [
              jsx('span', { className: 'font-medium', children: selectedCombination.profile }),
              jsx('span', { children: `${selectedCombination.main_quantization} main · ${selectedCombination.aux_quantization} auxiliary` }),
              jsx('span', { children: `${selectedCombination.validation_required ? 'benchmark required' : `${selectedCombination.min_tps} tok/s`} · ${selectedCombination.aux_mode} · ${selectedCombination.confidence}` }),
              jsx('span', { children: selectedCombination.intelligence_score == null ? 'Intelligence pending' : `Intelligence ${selectedCombination.intelligence_score} · ${selectedCombination.intelligence_level}` }),
              jsx('span', { style: { color: selectedCombination.fit ? 'var(--ui-text-tertiary)' : 'var(--ui-error)' }, children: selectedCombination.fit_reason }),
            ] }),
          }) : null,
          jsx(Field, {
            label: 'Provider endpoint',
            children: jsx('input', {
              value: baseUrl,
              onChange: (event) => setBaseUrl(event.target.value),
              style: fieldStyle,
            }),
          }),
          jsxs('div', { className: 'flex flex-col gap-2', children: [
            jsxs('label', { className: 'flex items-center gap-2', children: [
              jsx('input', { type: 'checkbox', checked: primary, onChange: (event) => setPrimary(event.target.checked) }),
              'Use Turbofit as primary provider',
            ] }),
            jsxs('label', { className: 'flex items-center gap-2', children: [
              jsx('input', { type: 'checkbox', checked: fallback, onChange: (event) => setFallback(event.target.checked) }),
              'Include Turbofit in fallback chain',
            ] }),
          ] }),
        ],
      }),
      jsxs('section', { className: 'flex flex-col gap-3', children: [
        jsx('h2', { className: 'font-semibold', children: 'Subscriptions and keyless fallbacks' }),
        jsx('p', { style: { color: 'var(--ui-text-tertiary)' }, children: 'Your configured providers first. Nous keyless free models last. Not a JSON dump.' }),
        jsx('div', { className: 'font-medium', children: 'Your subscriptions' }),
        ...(((status && status.catalog && status.catalog.subscriptions) || []).length
          ? (status.catalog.subscriptions.map((item) => jsxs('label', {
            key: item.id,
            className: 'flex items-center gap-2',
            children: [
              jsx('span', { children: item.label }),
              jsx('span', { style: { color: 'var(--ui-text-tertiary)' }, children: item.configured ? 'configured' : 'not configured' }),
            ],
          })))
          : [jsx('div', { key: 'no-subs', style: { color: 'var(--ui-text-tertiary)' }, children: 'No paid or OAuth providers in this Hermes config yet.' })]),
        jsx('div', { className: 'font-medium', children: 'Nous keyless free' }),
        ...(((status && status.catalog && status.catalog.nous_free) || []).map((item) => {
          const enabled = fallbackChain.some((row) => row.provider === 'nous' && row.model === item.model)
          return jsxs('label', {
            key: item.model,
            className: 'flex items-center gap-2',
            children: [
              jsx('input', {
                type: 'checkbox',
                checked: enabled,
                onChange: (event) => {
                  if (event.target.checked) setFallbackChain([...fallbackChain, { provider: 'nous', model: item.model }])
                  else setFallbackChain(fallbackChain.filter((row) => !(row.provider === 'nous' && row.model === item.model)))
                },
              }),
              jsx('span', { children: item.label }),
              jsx('span', { style: { color: 'var(--ui-text-tertiary)' }, children: item.model }),
            ],
          })
        })),
      ] }),
      jsxs('section', { className: 'flex flex-col gap-2', children: [
        jsx('h2', { className: 'font-semibold', children: 'Runtime and remote access' }),
        jsx('p', {
          style: { color: 'var(--ui-text-tertiary)' },
          children: 'These are the same setup actions exposed by Hermes Dashboard. Install actions run only when selected.',
        }),
        jsxs('label', { className: 'flex items-center gap-2', children: [
          jsx('input', { type: 'checkbox', checked: installNative, onChange: (event) => setInstallNative(event.target.checked) }),
          'Install and activate the native runtime (downloads the selected model and starts local processes)',
        ] }),
        jsxs('label', { className: 'flex items-center gap-2', children: [
          jsx('input', { type: 'checkbox', checked: installFreeToken, onChange: (event) => setInstallFreeToken(event.target.checked) }),
          'Install FreeToken candidate (NVIDIA, CUDA 13+, text-only MoE)',
        ] }),
        jsxs('label', { className: 'flex items-center gap-2', children: [
          jsx('input', { type: 'checkbox', checked: installLemonade, onChange: (event) => setInstallLemonade(event.target.checked) }),
          'Install or verify Lemonade support',
        ] }),
        jsxs('label', { className: 'flex items-center gap-2', children: [
          jsx('input', { type: 'checkbox', checked: installSirvir, onChange: (event) => setInstallSirvir(event.target.checked) }),
          'Install or update GitHub-current Sirvir customer service',
        ] }),
        jsxs('label', { className: 'flex items-center gap-2', children: [
          jsx('input', { type: 'checkbox', checked: publishTailnet, onChange: (event) => setPublishTailnet(event.target.checked) }),
          'Publish provider and dashboard privately with Tailscale Serve',
        ] }),
        publishTailnet ? jsx('div', {
          className: 'grid grid-cols-1 gap-3 md:grid-cols-2',
          children: [
            ['Dashboard local port', dashboardLocalPort, setDashboardLocalPort],
            ['Provider local port', providerLocalPort, setProviderLocalPort],
            ['Dashboard Tailnet HTTPS port', dashboardHttpsPort, setDashboardHttpsPort],
            ['Provider Tailnet HTTPS port', providerHttpsPort, setProviderHttpsPort],
          ].map(([label, value, setter]) => jsx(Field, {
            key: label,
            label,
            children: jsx('input', {
              type: 'number',
              value,
              onChange: (event) => setter(event.target.value),
              style: fieldStyle,
            }),
          })),
        }) : null,
      ] }),
      jsxs('section', { className: 'flex flex-col gap-2', children: [
        jsx('h2', { className: 'font-semibold', children: 'Multimodal models' }),
        jsx('p', {
          style: { color: 'var(--ui-text-tertiary)' },
          children: 'Fit uses total system and accelerator memory. Candidate adapters stay explicitly labeled until installed.',
        }),
        jsx('div', {
          className: 'grid grid-cols-1 gap-3 md:grid-cols-2',
          children: ['image', 'video', 'music', 'tts', 'stt'].map((modality) => jsx(Field, {
            key: modality,
            label: modality.toUpperCase(),
            children: jsxs('select', {
              value: multimodalSelections[modality] || '',
              onChange: (event) => setMultimodalSelections({ ...multimodalSelections, [modality]: event.target.value }),
              style: fieldStyle,
              children: [
                jsx('option', { value: '', children: 'Not selected' }),
                ...(((multimodal && multimodal.modalities && multimodal.modalities[modality]) || []).map((item) =>
                  jsx('option', {
                    key: item.id,
                    value: item.id,
                    children: `${item.name} · ${item.action} · ${item.fit ? 'fits' : 'does not fit'}`,
                  })
                )),
              ],
            }),
          })),
        }),
      ] }),
      jsxs('div', { className: 'flex gap-2', children: [
        jsx('button', { type: 'button', onClick: save, disabled: busy, style: buttonStyle, children: busy ? 'Working…' : 'Apply' }),
        jsx('button', { type: 'button', onClick: refresh, disabled: busy, style: buttonStyle, children: 'Run TurboFit Check' }),
      ] }),
      jsxs('section', { className: 'flex flex-col gap-2', children: [
        jsx('h2', { className: 'font-semibold', children: `${preference} recommendations` }),
        choices.length ? choices.map((item, index) => jsxs('div', {
          key: item.profile,
          style: { border: '1px solid var(--ui-stroke-secondary)', borderRadius: '6px', padding: '9px' },
          children: [
            jsx('div', { className: 'font-medium', children: `${index + 1}. ${item.profile}` }),
            jsx('div', {
              style: { color: 'var(--ui-text-tertiary)' },
              children: `${item.context} ctx · ${item.min_tps} tok/s · ${item.aux_mode} · ${item.confidence}`,
            }),
            jsxs('div', {
              className: 'grid grid-cols-1 gap-2 md:grid-cols-3',
              children: [
                jsx(ScoreBar, { label: 'Context', value: item.context, max: 1048576 }),
                jsx(ScoreBar, { label: 'tok/s', value: item.min_tps, max: 80 }),
                jsx(ScoreBar, { label: 'Confidence', value: item.confidence === 'measured' ? 100 : item.confidence === 'portable-fit' ? 55 : 25, max: 100 }),
              ],
            }),
            jsx('button', {
              type: 'button',
              disabled: busy,
              style: { ...buttonStyle, marginTop: '6px' },
              onClick: () => runShift(item.profile),
              children: 'Shift to this combination',
            }),
          ],
        })) : jsx('div', {
          style: { color: 'var(--ui-text-tertiary)' },
          children: 'No evidence-backed configuration currently fits this hardware.',
        }),
      ] }),
      compatibleLanes.length ? jsxs('section', { className: 'flex flex-col gap-2', children: [
        jsx('h2', { className: 'font-semibold', children: 'Compatible local lanes — same Fit List only' }),
        jsx('div', {
          style: { color: 'var(--ui-text-tertiary)' },
          children: 'Only Maple or Ornith on 8 GB VRAM. A dense 27B that spills into RAM is not a lane — it will crawl.',
        }),
        ...compatibleLanes.map((item) => jsxs('div', {
          key: item.profile,
          style: { border: '1px solid var(--ui-stroke-secondary)', borderRadius: '6px', padding: '9px' },
          children: [
            jsx('div', { className: 'font-medium', children: item.profile }),
            jsx('div', { style: { color: 'var(--ui-text-tertiary)' }, children: `${item.context} ctx · ${item.aux_mode} · ${item.confidence}` }),
            jsx('div', { style: { color: 'var(--ui-text-quaternary)' }, children: item.fit_reason }),
          ],
        })),
      ] }) : null,
      smoke ? jsxs('section', { className: 'flex flex-col gap-2', children: [
        jsx('h2', { className: 'font-semibold', children: 'Local smoke results' }),
        jsx('p', { style: { color: 'var(--ui-text-tertiary)' }, children: 'Health check of the currently serving loopback gateway. This is not a promotion benchmark and does not rank models.' }),
        jsx('div', { style: { border: '1px solid var(--ui-stroke-secondary)', borderRadius: '6px', padding: '8px' }, children: `${smoke.status || 'unknown'} · suite ${smoke.suite || 'local-runtime-smoke-v1'}` }),
        smoke.evidence_path ? jsx('div', { style: { color: 'var(--ui-text-quaternary)' }, children: smoke.evidence_path }) : null,
        (smoke.resource_warnings || []).length ? jsx('div', { style: { color: 'var(--ui-text-tertiary)' }, children: `Warnings: ${(smoke.resource_warnings || []).join(', ')}` }) : null,
      ] }) : null,
      audition ? jsxs('section', { className: 'flex flex-col gap-2', children: [
        jsx('h2', { className: 'font-semibold', children: 'Engine audition' }),
        jsx('p', { style: { color: 'var(--ui-text-tertiary)' }, children: `Pair ${audition.main} / ${audition.aux} @ ${audition.context}. Maple GGUF is fork llama.cpp or TurboHaul-on-that-fork. vLLM/SGLang need HF weights, not Maple TQ2_0.` }),
        ...((audition.engines || []).map((item) => jsxs('div', {
          key: item.engine_id,
          style: { border: '1px solid var(--ui-stroke-secondary)', borderRadius: '6px', padding: '9px' },
          children: [
            jsx('div', { className: 'font-medium', children: `${item.display_name} · ${item.audition}` }),
            jsx('div', { style: { color: 'var(--ui-text-tertiary)' }, children: item.serve_note || item.reason }),
          ],
        }))),
      ] }) : null,
      jsx('div', {
        style: { color: 'var(--ui-text-quaternary)' },
        children: `Gateway ${status && status.gateway && status.gateway.reachable ? 'online' : 'offline'} · exact recommendations use measured evidence; portable lanes require an on-box benchmark.`,
      }),
    ],
  })
}

export default {
  id: 'turbofit',
  name: 'Turbofit',
  register(ctx) {
    api = ctx
    ctx.registerMany([
      {
        id: 'page',
        area: ROUTES_AREA,
        data: { path: '/turbofit' },
        render: () => jsx(TurbofitPage, {}),
      },
      {
        id: 'nav',
        area: SIDEBAR_NAV_AREA,
        data: { path: '/turbofit', label: 'Turbofit', codicon: 'dashboard' },
      },
      {
        id: 'open',
        area: PALETTE_AREA,
        data: {
          id: 'turbofit.open',
          label: 'Open Turbofit',
          run: () => host.navigate('/turbofit'),
        },
      },
      {
        id: 'shift-up',
        area: PALETTE_AREA,
        data: {
          id: 'turbofit.shift-up',
          label: 'Turbofit: shift up',
          run: () => host.navigate('/turbofit'),
        },
      },
      {
        id: 'update',
        area: PALETTE_AREA,
        data: {
          id: 'turbofit.update',
          label: 'Turbofit: update plugin and Sirvir',
          run: () => host.navigate('/turbofit'),
        },
      },
    ])
  },
}
