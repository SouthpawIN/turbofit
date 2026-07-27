(function () {
  "use strict";
  const SDK = window.__HERMES_PLUGIN_SDK__;
  const Registry = window.__HERMES_PLUGINS__;
  if (!SDK || !Registry) return;

  const React = SDK.React;
  const h = React.createElement;
  const { useCallback, useEffect, useMemo, useState } = SDK.hooks;
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
    const [profiles, setProfiles] = useState([]);
    const [primary, setPrimary] = useState(false);
    const [fallback, setFallback] = useState(false);
    const [profile, setProfile] = useState("auto");
    const [baseUrl, setBaseUrl] = useState("http://127.0.0.1:8091/v1");
    const [publishTailnet, setPublishTailnet] = useState(false);
    const [dashboardLocalPort, setDashboardLocalPort] = useState("9127");
    const [providerLocalPort, setProviderLocalPort] = useState("8091");
    const [dashboardHttpsPort, setDashboardHttpsPort] = useState("9444");
    const [providerHttpsPort, setProviderHttpsPort] = useState("9443");
    const [busy, setBusy] = useState(false);
    const [message, setMessage] = useState("");

    const load = useCallback(async function () {
      setMessage("");
      try {
        const [nextStatus, nextProfiles] = await Promise.all([
          SDK.fetchJSON("/api/plugins/turbofit/status"),
          SDK.fetchJSON("/api/plugins/turbofit/profiles"),
        ]);
        setStatus(nextStatus);
        setProfiles(nextProfiles.profiles || []);
        setPrimary(Boolean(value(nextStatus, "provider.primary", false)));
        setFallback(Boolean(value(nextStatus, "provider.fallback", false)));
        setBaseUrl(value(nextStatus, "provider.base_url", "http://127.0.0.1:8091/v1"));
        setProfile(value(nextStatus, "selection.requested", "auto"));
      } catch (error) {
        setMessage(String(error.message || error));
      }
    }, []);

    useEffect(function () { load(); }, [load]);

    const compatible = useMemo(function () {
      return profiles.filter((item) => item.manual_compatible);
    }, [profiles]);

    async function save() {
      setBusy(true);
      setMessage("");
      try {
        const result = await SDK.fetchJSON("/api/plugins/turbofit/configure", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            primary, fallback, profile, base_url: baseUrl,
            publish_tailnet: publishTailnet,
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
        ),
      ),

      message ? h("div", { className: "tf-message" }, message) : null,

      h("div", { className: "tf-grid" },
        h(Card, { className: "tf-card tf-span-2" },
          h(CardHeader, null, h(CardTitle, null, "Setup")),
          h(CardContent, { className: "tf-form" },
            h("div", { className: "tf-two" },
              h(Field, {
                id: "tf-profile", label: "Hardware profile",
                help: "Auto scans physical topology. Manual choices are limited to compatible profiles.",
              },
                h(Select, { id: "tf-profile", value: profile, onChange: (event) => setProfile(event.target.value) },
                  h(SelectOption, { value: "auto" }, "Auto — recommended"),
                  compatible.map((item) => h(SelectOption, { key: item.id, value: item.id },
                    item.id + " · " + item.topology + " · " + item.rungs.length + " rungs"
                  )),
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
            ),
            h("div", { className: "tf-actions" },
              h(Button, { onClick: save, disabled: busy }, busy ? "Applying…" : "Apply configuration"),
              h(Button, { variant: "outline", onClick: load, disabled: busy }, "Refresh"),
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
      ),

      h("p", { className: "tf-footnote" },
        "Turbofit never terminates external GPU processes. Configuration changes are atomic; provider changes take effect in a new Hermes session."
      ),
    );
  }

  Registry.register("turbofit", TurbofitPage);
})();
