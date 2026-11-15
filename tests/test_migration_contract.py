"""Controller-held integration regressions; fixtures do not run model inference."""
import json
from pathlib import Path
import pytest
from turbofit_runtime.recipes import LAUNCH_OVERRIDE_FLAGS
from turbofit_runtime.request_normalisation import normalise_reasoning_budget
from turbofit_runtime.routes import load_runtime_resolutions
from test_gateway_compatibility import GATEWAY, fake_stack, post
from test_native_backend import backend


def test_override_flag_names_match_preserved_engine_parser():
    assert LAUNCH_OVERRIDE_FLAGS['checkpoints'] == ('--ctx-checkpoints',)
    assert LAUNCH_OVERRIDE_FLAGS['slot_similarity'] == ('--slot-prompt-similarity',)


def test_role_context_metadata_round_trips_resolution_loader(tmp_path):
    p = tmp_path/'roles.json'
    p.write_text(json.dumps({'schema':'turbofit.runtime-resolutions/v1', 'profiles':{
        'p':{'r':{'main':{'model_tag':'main','expected_vram_mb':1,'context':204800},
                  'aux':{'model_tag':'aux','expected_vram_mb':1,'context':262144}}}}}))
    roles=load_runtime_resolutions(p)['p']['r']
    assert roles['main']['context']==204800
    assert roles['aux']['context']==262144


def test_native_component_uses_role_context(tmp_path):
    runtime=backend(tmp_path)
    item=dict(runtime._roles('local-gsq-rco-iq3s-262144')['main'],context=131072)
    cmd=runtime._component('main',item,262144).command
    assert cmd[cmd.index('-c')+1]=='131072'


def test_gateway_advertises_independent_contexts(tmp_path,monkeypatch):
    p=tmp_path/'routes.json'
    p.write_text(json.dumps({'active':'private','routes':{
        'main':{'kind':'local','context_length':204800},
        'aux':{'kind':'local','context_length':262144}}}))
    monkeypatch.setattr(GATEWAY,'RUNTIME_STATE',str(p))
    monkeypatch.setattr(GATEWAY,'_live_n_ctx',lambda _: (_ for _ in ()).throw(AssertionError('unnecessary probe')))
    got={x['id']:x['context_length'] for x in GATEWAY.provider_models()}
    assert got=={'auto':204800}


@pytest.mark.parametrize('policy',[{'default_budget':1024},{'hard_cap':0},{'hard_cap':True},{'hard_cap':'bad'}])
def test_configured_budget_policy_cannot_disable_bound(policy):
    with pytest.raises(ValueError):
        normalise_reasoning_budget({'thinking_budget_tokens':2147483647},policy)


def test_normalisation_failure_is_not_swallowed(fake_stack,monkeypatch):
    client,captured=fake_stack
    monkeypatch.setattr(GATEWAY,'reasoning_policy_for',lambda _: {'hard_cap':0})
    response=post(client,{'model':'active:main','messages':[],'thinking_budget_tokens':2147483647})
    assert response.status==400
    response.read()
    assert 'payload' not in captured


def test_main_thinking_default_remains_configurable(fake_stack,monkeypatch):
    client,captured=fake_stack
    monkeypatch.setattr(GATEWAY,'MAIN_ENABLE_THINKING',False)
    response=post(client,{'model':'active:main','messages':[]})
    assert response.status==200
    response.read()
    assert captured['payload']['chat_template_kwargs']['enable_thinking'] is False
