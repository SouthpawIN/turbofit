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
  const [fallbackChain, setFallbackChain] = useState('[]')
  const [publishTailnet, setPublishTailnet] = useState(false)
  const [installSirvir, setInstallSirvir] = useState(false)
  const [installNative, setInstallNative] = useState(false)
  const [installLemonade, setInstallLemonade] = useState(false)
  const [dashboardLocalPort, setDashboardLocalPort] = useState('9127')
  const [providerLocalPort, setProviderLocalPort] = useState('8091')
  const [dashboardHttpsPort, setDashboardHttpsPort] = useState('9444')
  const [providerHttpsPort, setProviderHttpsPort] = useState('9443')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')

  const refresh = useCallback(async () => {
    setBusy(true)
    setMessage('')
    try {
      const [nextStatus, nextCombinations, nextRecommendations, nextMultimodal] = await Promise.all([
        api.rest('/status'),
        api.rest('/combinations'),
        api.rest(`/recommendations?preference=${encodeURIComponent(preference)}`),
        api.rest('/multimodal'),
      ])
      setStatus(nextStatus)
      const exact = (nextCombinations && nextCombinations.combinations) || []
      setCombinations(exact)
      setRecommendations(nextRecommendations)
      setMultimodal(nextMultimodal)
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
      setFallbackChain(JSON.stringify(provider.fallback_chain || [], null, 2))
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
      const chain = fallbackChain.trim() ? JSON.parse(fallbackChain) : []
      if (!Array.isArray(chain)) throw new Error('Fallback chain must be a JSON array.')
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

  const choices = ((recommendations && recommendations.recommendations) || {})[preference] || []
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
      message ? jsx('div', {
        style: { border: '1px solid var(--ui-stroke-secondary)', borderRadius: '6px', padding: '8px' },
        children: message,
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
            help: 'Auto chooses the best evidence-backed fit. Manual exposes every exact validated combination.',
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
              jsx('span', { children: `${selectedCombination.min_tps} tok/s · ${selectedCombination.aux_mode} · ${selectedCombination.confidence}` }),
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
      jsx(Field, {
        label: 'Fallback chain',
        help: 'Ordered provider/model JSON. Keep credentials in Hermes provider configuration.',
        children: jsx('textarea', {
          value: fallbackChain,
          onChange: (event) => setFallbackChain(event.target.value),
          rows: 7,
          spellCheck: false,
          style: { ...fieldStyle, fontFamily: 'var(--ui-font-mono)', resize: 'vertical' },
        }),
      }),
      jsxs('section', { className: 'flex flex-col gap-2', children: [
        jsx('h2', { className: 'font-semibold', children: 'Runtime and remote access' }),
        jsx('p', {
          style: { color: 'var(--ui-text-tertiary)' },
          children: 'These are the same setup actions exposed by Hermes Dashboard. Install actions run only when selected.',
        }),
        jsxs('label', { className: 'flex items-center gap-2', children: [
          jsx('input', { type: 'checkbox', checked: installNative, onChange: (event) => setInstallNative(event.target.checked) }),
          'Install or verify the native adaptive runtime',
        ] }),
        jsxs('label', { className: 'flex items-center gap-2', children: [
          jsx('input', { type: 'checkbox', checked: installLemonade, onChange: (event) => setInstallLemonade(event.target.checked) }),
          'Install or verify Lemonade support',
        ] }),
        jsxs('label', { className: 'flex items-center gap-2', children: [
          jsx('input', { type: 'checkbox', checked: installSirvir, onChange: (event) => setInstallSirvir(event.target.checked) }),
          'Install or update bundled Sirvir customer service',
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
        jsx('button', { type: 'button', onClick: refresh, disabled: busy, style: buttonStyle, children: 'Rescan' }),
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
          ],
        })) : jsx('div', {
          style: { color: 'var(--ui-text-tertiary)' },
          children: 'No evidence-backed configuration currently fits this hardware.',
        }),
      ] }),
      jsx('div', {
        style: { color: 'var(--ui-text-quaternary)' },
        children: `Gateway ${status && status.gateway && status.gateway.reachable ? 'online' : 'offline'} · recommendations use physical capacity, measured evidence, and current backend support.`,
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
    ])
  },
}
