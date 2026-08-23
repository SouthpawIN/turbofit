(function () {
  "use strict";
  const SDK = window.__HERMES_PLUGIN_SDK__;
  const Registry = window.__HERMES_PLUGINS__;
  if (!SDK || !Registry) return;

  const React = SDK.React;
  const h = React.createElement;
  const { useCallback, useEffect, useState } = SDK.hooks;
  const {
    Badge, Button, Card, CardContent, CardHeader, CardTitle,
    Input, Label, Select, SelectOption, Separator,
  } = SDK.components;

  function value(obj, path, fallback) {
    let current = obj;
    for (const key of path.split(".")) {
      if (!current || typeof current !== "object") return fallback;
      current = current[key];
    }
    return current == null ? fallback : current;
  }

  function statusBadge(ok, yes, no) {
    return h(Badge, { className: ok ? "tf-ok" : "tf-muted" }, ok ? yes : no);
  }

  function Field(props) {
    return h("div", { className: "tf-field" },
      h(Label, { htmlFor: props.id }, props.label),
      props.children,
      props.help ? h("p", { className: "tf-help" }, props.help) : null,
    );
  }

  function Toggle(props) {
    return h("label", { className: "tf-toggle" },
      h("input", {
        type: "checkbox",
        checked: props.checked,
        onChange: (event) => props.onChange(event.target.checked),
      }),
      h("span", null, props.label),
    );
  }

  function TurbofitPage() {
    const [status, setStatus] = useState(null);
    const [backends, setBackends] = useState(null);
    const [combinations, setCombinations] = useState([]);
    const [tournaments, setTournaments] = useState(null);
    const [hardwareTiers, setHardwareTiers] = useState(null);
    const [auxiliaryTiers, setAuxiliaryTiers] = useState(null);
    const [multimodal, setMultimodal] = useState(null);
    const [multimodalSelections, setMultimodalSelections] = useState({});
    const [primary, setPrimary] = useState(false);
    const [fallback, setFallback] = useState(false);
    const [fallbackChainText, setFallbackChainText] = useState("[]");
    const [selectionMode, setSelectionMode] = useState("auto");
    const [mainModel, setMainModel] = useState("");
    const [auxModel, setAuxModel] = useState("");
    const [context, setContext] = useState("");
    const [servingBackend, setServingBackend] = useState("adaptive");
    const [baseUrl, setBaseUrl] = useState("http://127.0.0.1:8091/v1");
    const [publishTailnet, setPublishTailnet] = useState(false);
    const [installSirvir, setInstallSirvir] = useState(true);
    const [installDesktop, setInstallDesktop] = useState(true);
    const [installLemonade, setInstallLemonade] = useState(false);
    const [installNative, setInstallNative] = useState(false);
    const [dashboardLocalPort, setDashboardLocalPort] = useState("9127");
    const [providerLocalPort, setProviderLocalPort] = useState("8091");
    const [dashboardHttpsPort, setDashboardHttpsPort] = useState("9444");
    const [providerHttpsPort, setProviderHttpsPort] = useState("9443");
    const [busy, setBusy] = useState(false);
    const [message, setMessage] = useState("");

    const load = useCallback(async function () {
      setMessage("");
      try {
        const [nextStatus, nextCombinations, nextBackends, nextTournaments, nextHardwareTiers, nextAuxiliaryTiers, nextMultimodal] = await Promise.all([
          SDK.fetchJSON("/api/plugins/turbofit/status"),
          SDK.fetchJSON("/api/plugins/turbofit/combinations"),
          SDK.fetchJSON("/api/plugins/turbofit/backends"),
          SDK.fetchJSON("/api/plugins/turbofit/tournaments"),
          SDK.fetchJSON("/api/plugins/turbofit/hardware-tiers"),
          SDK.fetchJSON("/api/plugins/turbofit/auxiliary-tiers"),
          SDK.fetchJSON("/api/plugins/turbofit/multimodal"),
        ]);
        setStatus(nextStatus);
        const exact = nextCombinations.combinations || [];
        setCombinations(exact);
        setBackends(nextBackends);
        setTournaments(nextTournaments);
        setHardwareTiers(nextHardwareTiers);
        setAuxiliaryTiers(nextAuxiliaryTiers);
        setMultimodal(nextMultimodal);
        const selectedModalities = Object.assign({}, nextMultimodal.selected || {});
        ["image", "video", "music", "tts", "stt"].forEach((modality) => {
          if (!selectedModalities[modality]) {
            const choice = (value(nextMultimodal, "modalities." + modality, [])).find((item) => item.recommended_fit)
              || (value(nextMultimodal, "modalities." + modality, [])).find((item) => item.fit);
            if (choice) selectedModalities[modality] = choice.id;
          }
        });
        setMultimodalSelections(selectedModalities);
        setPrimary(Boolean(value(nextStatus, "provider.primary", false)));
        setFallback(Boolean(value(nextStatus, "provider.fallback", false)));
        setFallbackChainText(JSON.stringify(value(nextStatus, "provider.fallback_chain", []), null, 2));
        setBaseUrl(value(nextStatus, "provider.base_url", "http://127.0.0.1:8091/v1"));
        setServingBackend(value(nextStatus, "provider.base_url", "").includes(":13305/") ? "lemonade" : "adaptive");
        const requested = value(nextStatus, "selection.requested", "auto");
        const exactId = requested.startsWith("manual-") ? requested.slice(7) : requested;
        const selected = exact.find((item) => item.profile === exactId);
        if (selected) {
          setSelectionMode("manual");
          setMainModel(selected.main);
          setAuxModel(selected.aux);
          setContext(String(selected.context));
        } else {
          setSelectionMode("auto");
          const first = exact.find((item) => item.fit) || exact[0];
          if (first) {
            setMainModel(first.main);
            setAuxModel(first.aux);
            setContext(String(first.context));
          }
        }
      } catch (error) {
        setMessage(String(error.message || error));
      }
    }, []);

    useEffect(function () { load(); }, [load]);

    const mainModels = [...new Set(combinations.map((item) => item.main))];
    const mainRows = combinations.filter((item) => item.main === mainModel);
    const auxModels = [...new Set(mainRows.map((item) => item.aux))];
    const pairRows = mainRows.filter((item) => item.aux === auxModel);
    const contexts = [...new Set(pairRows.map((item) => item.context))].sort((a, b) => a - b);
    const selectedCombination = pairRows.find((item) => String(item.context) === String(context));

    async function save() {
      setBusy(true);
      setMessage("");
      try {
        const fallbackChain = fallbackChainText.trim() ? JSON.parse(fallbackChainText) : [];
        if (!Array.isArray(fallbackChain)) throw new Error("Fallback chain must be a JSON array.");
        if (selectionMode === "manual" && !selectedCombination) throw new Error("Choose an exact main, auxiliary, and context combination.");
        if (selectionMode === "manual" && !selectedCombination.fit) throw new Error(selectedCombination.fit_reason || "This combination does not fit this machine.");
        const result = await SDK.fetchJSON("/api/plugins/turbofit/configure", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            primary, fallback, fallback_chain: fallbackChain, multimodal: multimodalSelections,
            profile: selectionMode === "auto" ? "auto" : selectedCombination.profile, base_url: baseUrl,
            publish_tailnet: publishTailnet,
            install_sirvir: installSirvir,
            install_desktop: installDesktop,
            install_lemonade: installLemonade,
            install_native: installNative,
            dashboard_local_port: Number(dashboardLocalPort),
            provider_local_port: Number(providerLocalPort),
            dashboard_https_port: Number(dashboardHttpsPort),
            provider_https_port: Number(providerHttpsPort),
          }),
        });
        setMessage(result.restart_required
          ? "Saved. Start a new Hermes session to use the provider changes."
          : "Saved.");
        await load();
      } catch (error) {
        setMessage(String(error.message || error));
      } finally {
        setBusy(false);
      }
    }

    const gateway = value(status, "gateway", {});
    const tailnet = value(status, "tailnet", {});
    const runtime = value(status, "runtime", null);
    const selection = value(status, "selection", null);
    const routes = value(runtime, "routes", {});

    return h("div", { className: "tf-page" },
      h("div", { className: "tf-hero" },
        h("div", null,
          h("p", { className: "tf-kicker" }, "LOCAL INFERENCE · ADAPTIVE POLICY"),
          h("h1", null, "Turbofit"),
          h("p", { className: "tf-subtitle" },
            "One stable Hermes provider that yields VRAM to your work, then heals back to the best proven configuration."
          ),
        ),
        h("div", { className: "tf-hero-status" },
          statusBadge(Boolean(gateway.reachable), "Gateway online", "Gateway offline"),
          statusBadge(Boolean(tailnet.connected), "Tailnet connected", "Tailnet unavailable"),
          statusBadge(Boolean(value(status, "provider.registered", false)), "Provider registered", "Not registered"),
          statusBadge(Boolean(value(backends, "lemonade.available", false)), "Lemonade online", "Lemonade optional"),
        ),
      ),

      message ? h("div", { className: "tf-message" }, message) : null,

      h("div", { className: "tf-grid" },
        h(Card, { className: "tf-card tf-span-2" },
          h(CardHeader, null, h(CardTitle, null, "Setup")),
          h(CardContent, { className: "tf-form" },
            h("div", { className: "tf-two" },
              h(Field, {
                id: "tf-selection-mode", label: "Runtime selection",
                help: "Auto chooses an exact-hardware winner. Manual exposes physical-fit lanes and marks candidates that require an on-box benchmark.",
              },
                h(Select, { id: "tf-selection-mode", value: selectionMode, onChange: (event) => setSelectionMode(event.target.value) },
                  h(SelectOption, { value: "auto" }, "Auto — recommended"),
                  h(SelectOption, { value: "manual" }, "Manual — " + combinations.filter((item) => item.fit).length + " compatible combinations"),
                ),
              ),
              selectionMode === "manual" ? h(Field, { id: "tf-main-model", label: "Main model" },
                h(Select, {
                  id: "tf-main-model", value: mainModel,
                  onChange: (event) => {
                    const selectedMain = event.target.value;
                    const rows = combinations.filter((item) => item.main === selectedMain);
                    const first = rows.find((item) => item.fit) || rows[0];
                    setMainModel(selectedMain);
                    setAuxModel(first ? first.aux : "");
                    setContext(first ? String(first.context) : "");
                  },
                },
                  mainModels.map((model) => {
                    const rows = combinations.filter((item) => item.main === model);
                    const fits = combinations.some((item) => item.main === model && item.fit);
                    const label = rows[0] ? rows[0].main_name + " · " + rows[0].main_quantization : model;
                    return h(SelectOption, { key: model, value: model, disabled: !fits }, label + (fits ? "" : " — unavailable"));
                  }),
                ),
              ) : null,
              selectionMode === "manual" ? h(Field, { id: "tf-aux-model", label: "Auxiliary model" },
                h(Select, {
                  id: "tf-aux-model", value: auxModel,
                  onChange: (event) => {
                    const selectedAux = event.target.value;
                    const rows = mainRows.filter((item) => item.aux === selectedAux);
                    const first = rows.find((item) => item.fit) || rows[0];
                    setAuxModel(selectedAux);
                    setContext(first ? String(first.context) : "");
                  },
                },
                  auxModels.map((model) => {
                    const rows = mainRows.filter((item) => item.aux === model);
                    const fits = mainRows.some((item) => item.aux === model && item.fit);
                    const label = rows[0] ? rows[0].aux_name + " · " + rows[0].aux_quantization : model;
                    return h(SelectOption, { key: model, value: model, disabled: !fits }, label + (fits ? "" : " — unavailable"));
                  }),
                ),
              ) : null,
              selectionMode === "manual" ? h(Field, { id: "tf-context", label: "Context length" },
                h(Select, { id: "tf-context", value: context, onChange: (event) => setContext(event.target.value) },
                  contexts.map((contextValue) => {
                    const row = pairRows.find((item) => item.context === contextValue);
                    return h(SelectOption, { key: contextValue, value: String(contextValue), disabled: !row.fit },
                      contextValue.toLocaleString() + " tokens" + (row.fit ? "" : " — does not fit")
                    );
                  }),
                ),
              ) : null,
              selectionMode === "manual" && selectedCombination ? h("div", { className: "tf-combination" },
                h("strong", null, selectedCombination.profile),
                h("p", { className: "tf-help" }, selectedCombination.main_quantization + " main · " + selectedCombination.aux_quantization + " auxiliary"),
                h("p", { className: "tf-help" },
                  selectedCombination.validation_required
                    ? "benchmark required · " + selectedCombination.aux_mode + " · " + selectedCombination.confidence
                    : selectedCombination.min_tps + " tok/s · " + selectedCombination.aux_mode + " · " + selectedCombination.confidence
                ),
                h("p", { className: "tf-help" }, selectedCombination.fit_reason),
              ) : null,
              h(Field, {
                id: "tf-serving-backend", label: "Serving backend",
                help: "Adaptive Turbofit is the default. Lemonade uses its local OpenAI-compatible server.",
              },
                h(Select, {
                  id: "tf-serving-backend", value: servingBackend,
                  onChange: (event) => {
                    const selected = event.target.value;
                    setServingBackend(selected);
                    setInstallLemonade(selected === "lemonade");
                    setInstallNative(selected === "adaptive");
                    setBaseUrl(selected === "lemonade"
                      ? "http://127.0.0.1:13305/api/v1"
                      : "http://127.0.0.1:8091/v1");
                  },
                },
                  h(SelectOption, { value: "adaptive" }, "Turbofit adaptive runtime"),
                  h(SelectOption, { value: "lemonade" }, "Lemonade Server"),
                ),
              ),
              h(Field, {
                id: "tf-endpoint", label: "Provider endpoint",
                help: "Use localhost or an HTTPS/Tailscale URL ending in /v1.",
              },
                h(Input, {
                  id: "tf-endpoint", value: baseUrl,
                  onChange: (event) => setBaseUrl(event.target.value),
                  placeholder: "http://127.0.0.1:8091/v1",
                }),
              ),
            ),
            h(Separator, null),
            h("div", { className: "tf-toggles" },
              h(Toggle, {
                checked: publishTailnet,
                onChange: setPublishTailnet,
                label: "Publish provider and dashboard privately with Tailscale Serve",
              }),
            ),
            publishTailnet ? h("div", { className: "tf-two" },
              h(Field, { id: "tf-dashboard-local", label: "Dashboard local port" },
                h(Input, { id: "tf-dashboard-local", type: "number", value: dashboardLocalPort, onChange: (event) => setDashboardLocalPort(event.target.value) })
              ),
              h(Field, { id: "tf-dashboard-https", label: "Dashboard Tailnet HTTPS port" },
                h(Input, { id: "tf-dashboard-https", type: "number", value: dashboardHttpsPort, onChange: (event) => setDashboardHttpsPort(event.target.value) })
              ),
              h(Field, { id: "tf-provider-local", label: "Provider local port" },
                h(Input, { id: "tf-provider-local", type: "number", value: providerLocalPort, onChange: (event) => setProviderLocalPort(event.target.value) })
              ),
              h(Field, { id: "tf-provider-https", label: "Provider Tailnet HTTPS port" },
                h(Input, { id: "tf-provider-https", type: "number", value: providerHttpsPort, onChange: (event) => setProviderHttpsPort(event.target.value) })
              ),
            ) : null,
            h(Separator, null),
            h("div", { className: "tf-toggles" },
              h(Toggle, { checked: primary, onChange: setPrimary, label: "Use Turbofit as primary provider (model: auto)" }),
              h(Toggle, { checked: fallback, onChange: setFallback, label: "Add Turbofit to Hermes fallback providers" }),
              h(Toggle, { checked: installSirvir, onChange: setInstallSirvir, label: "Install or update GitHub-current Sirvir customer service" }),
              h(Toggle, { checked: installDesktop, onChange: setInstallDesktop, label: "Install native Hermes Desktop Turbofit page" }),
              servingBackend === "adaptive" ? h(Toggle, {
                checked: installNative,
                onChange: setInstallNative,
                label: "Build or check the pinned native llama.cpp runtime",
              }) : null,
              servingBackend === "lemonade" ? h(Toggle, {
                checked: installLemonade,
                onChange: setInstallLemonade,
                label: "Install or start the pinned Lemonade Server runtime",
              }) : null,
            ),
            h(Field, {
              id: "tf-fallback-chain", label: "Fallback chain",
              help: "Ordered Hermes provider/model entries. Edit the JSON directly; credentials stay in Hermes, never here.",
            },
              h("textarea", {
                id: "tf-fallback-chain",
                className: "tf-fallback-chain",
                value: fallbackChainText,
                onChange: (event) => setFallbackChainText(event.target.value),
                rows: 7,
                spellCheck: false,
              }),
            ),
            h("div", { className: "tf-actions" },
              h(Button, { onClick: save, disabled: busy }, busy ? "Applying…" : "Apply configuration"),
              h(Button, { variant: "outline", onClick: load, disabled: busy }, "Refresh"),
            ),
          ),
        ),

        h(Card, { className: "tf-card tf-span-2" },
          h(CardHeader, null, h(CardTitle, null, "Multimodal models")),
          h(CardContent, { className: "tf-form" },
            h("p", { className: "tf-help" },
              "Recommendations use total usable system and accelerator memory. Built-in options work now; candidate adapters remain labeled until installed and verified."
            ),
            h("div", { className: "tf-two" },
              ["image", "video", "music", "tts", "stt"].map((modality) =>
                h(Field, { key: modality, id: "tf-" + modality, label: modality.toUpperCase() },
                  h(Select, {
                    id: "tf-" + modality,
                    value: multimodalSelections[modality] || "",
                    onChange: (event) => setMultimodalSelections(Object.assign({}, multimodalSelections, { [modality]: event.target.value })),
                  },
                    h(SelectOption, { value: "" }, "Not selected"),
                    value(multimodal, "modalities." + modality, []).map((item) =>
                      h(SelectOption, { key: item.id, value: item.id },
                        item.name + " · " + item.action + " · " + (item.fit ? "fits" : "does not fit")
                      )
                    ),
                  ),
                )
              ),
            ),
          ),
        ),

        h(Card, { className: "tf-card" },
          h(CardHeader, null, h(CardTitle, null, "Active policy")),
          h(CardContent, null,
            h("dl", { className: "tf-stats" },
              h("div", null, h("dt", null, "Selection"), h("dd", null, value(selection, "requested", "Not selected"))),
              h("div", null, h("dt", null, "Profile"), h("dd", null, value(selection, "profile_id", "—"))),
              h("div", null, h("dt", null, "Rung"), h("dd", null, value(runtime, "rung_id", "—"))),
              h("div", null, h("dt", null, "State"), h("dd", null, value(runtime, "status", value(runtime, "phase", "—")))),
              h("div", null, h("dt", null, "Backend"), h("dd", null, value(backends, "local_backend", "—"))),
            ),
          ),
        ),

        h(Card, { className: "tf-card" },
          h(CardHeader, null, h(CardTitle, null, "Stable routes")),
          h(CardContent, null,
            h("div", { className: "tf-routes" },
              ["main", "aux"].map((role) => h("div", { key: role, className: "tf-route" },
                h("span", { className: "tf-route-name" }, "active:" + role),
                h("strong", null, value(routes, role + ".alias", value(routes, role + ".model_tag", "unpublished"))),
                h("small", null, value(routes, role + ".mode", value(routes, role + ".kind", "—"))),
              )),
            ),
          ),
        ),

        h(Card, { className: "tf-card tf-span-2" },
          h(CardHeader, null, h(CardTitle, null, "Hardware tiers · speed versus intelligence")),
          h(CardContent, null,
            h("p", { className: "tf-help" },
              "Scores appear only after DeepSWE and the agentic production-pair harness finish on the exact quantized recipe. Pending is not treated as zero."
            ),
            h("dl", { className: "tf-stats" },
              value(hardwareTiers, "tiers", []).map((tier) => {
                const winner = value(tier, "recommendations.measured_winner", null);
                const smart = value(tier, "recommendations.smartest", null);
                const fast = value(tier, "recommendations.fastest", null);
                const balanced = value(tier, "recommendations.balanced", null);
                const shown = smart || fast || winner;
                const intel = shown && shown.intelligence_score != null ? shown.intelligence_score.toFixed(1) : "pending";
                const tps = shown && shown.measured_tps != null ? shown.measured_tps.toFixed(2) : "pending";
                return h("div", { key: tier.id },
                  h("dt", null, tier.capacity_gb + " GB · " + value(tier, "topology.topology", "unknown")),
                  h("dd", null,
                    (shown ? shown.configuration_id : "no physical winner") +
                    " · intelligence " + intel + " · " + tps + " tok/s" +
                    (balanced ? " · balanced " + balanced.configuration_id : "")
                  ),
                );
              }),
            ),
          ),
        ),

        h(Card, { className: "tf-card tf-span-2" },
          h(CardHeader, null, h(CardTitle, null, "Hardware-tier tournaments")),
          h(CardContent, null,
            h("p", { className: "tf-help" },
              "A tier is promoted only from hash-bound evidence captured on its physical hardware topology."
            ),
            h("dl", { className: "tf-stats" },
              value(tournaments, "tiers", []).map((tier) => {
                const successful = tier.candidates.filter((item) => item.status === "success").length;
                const winner = tier.winner && tier.winner.configuration;
                return h("div", { key: tier.id },
                  h("dt", null, tier.vram_gb + " GB"),
                  h("dd", null, winner || (successful + "/" + tier.candidates.length + " tested · promotion pending")),
                );
              }),
            ),
          ),
        ),

        h(Card, { className: "tf-card tf-span-2" },
          h(CardHeader, null, h(CardTitle, null, "Auxiliary candidates by hardware tier")),
          h(CardContent, null,
            h("p", { className: "tf-help" },
              "No winner is shown until the exact current runtime recipe passes intelligence, TPS, context, fit, and exact-topology evidence gates."
            ),
            h("dl", { className: "tf-stats" },
              value(auxiliaryTiers, "tiers", []).map((tier) => h("div", { key: tier.vram_gb },
                h("dt", null, tier.vram_gb + " GB"),
                h("dd", null,
                  (tier.best_auxiliary || tier.candidate || "benchmarking") + " · " + tier.status
                ),
              )),
            ),
          ),
        ),
      ),

      h("p", { className: "tf-footnote" },
        "Turbofit never terminates external GPU processes. Configuration changes are atomic; provider changes take effect in a new Hermes session."
      ),
    );
  }

  Registry.register("turbofit", TurbofitPage);
})();
