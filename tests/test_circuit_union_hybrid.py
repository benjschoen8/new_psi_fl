import importlib
import importlib.util
from unittest.mock import patch

import pytest


def module():
    assert importlib.util.find_spec('label_union.circuit_union_hybrid'), 'new hybrid union is missing'
    return importlib.import_module('label_union.circuit_union_hybrid')


def test_missing_real_backend_is_not_simulated(monkeypatch):
    monkeypatch.delenv('MPSPDZ', raising=False)
    with pytest.raises(RuntimeError, match='MPSPDZ'):
        module().circuit_union_with_keys([['cat']] * 3, [{'cat': 'cat'}] * 3)


def test_hybrid_keys_drive_union_without_cleartext_grouping(monkeypatch):
    monkeypatch.setenv('MPSPDZ', '/mock-native-backend')
    labels = [['cat'], ['cat'], ['dog']]
    keywords = [{label: label for label in own} for own in labels]
    with patch('label_union.mpspdz_pairwise.mpspdz_pairwise_group',
               return_value=([0, 0, 2], {'global_MB': 1, 'prefix': 'serial'}, [7, 7, 9])) as backend, patch(
            'label_union.circuit_union.group', side_effect=AssertionError('plaintext oracle used')):
        indices, secrets, public_keys, size, stats = module().circuit_union_with_keys(
            labels, keywords, bucket_bits=8, mpc_options={'prefix': 'serial'})
    assert size == 2
    assert indices[0]['cat'] == indices[1]['cat'] != indices[2]['dog']
    assert secrets[0]['cat'] == secrets[1]['cat'] != secrets[2]['dog']
    assert len(public_keys) == 2
    assert backend.call_args.kwargs['prefix'] == 'serial'
    assert stats['mpc']['measured']['global_MB'] == 1
    assert stats['setup_upload_bytes_per_client'] is None
    assert stats['setup_download_bytes_per_client'] is None


def test_global_comparison_delegates_to_untouched_original():
    with patch('label_union.circuit_union.circuit_union_with_keys', return_value='original') as original:
        assert module().circuit_union_with_keys([['cat']] * 3, [{'cat': 'cat'}] * 3,
                    mpc_backend='global', mpc_options={'prefix': 'parallel'}) == 'original'
    assert 'mpc_options' not in original.call_args.kwargs
    assert 'mpc_backend' not in original.call_args.kwargs
