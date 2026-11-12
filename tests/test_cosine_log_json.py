"""MPI log prefixes can occur inside one long JSON scalar or string."""

import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('cosine_controller',
    Path(__file__).resolve().parents[1] / 'scripts/cloud/cosine_257m_controller.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


class LogFramingTest(unittest.TestCase):
    def test_multiple_mpi_fragments_inside_long_json(self):
        original = dict(path='/results/run/metrics.jsonl',
                        validation=[dict(iter=i * 500, loss=3.123456789, name='experiment')
                                    for i in range(100)])
        raw = json.dumps(original)
        framed = '[1,0]<stdout>:'.join(raw[i:i + 4089] for i in range(0, len(raw), 4089))
        self.assertNotEqual(json.loads(framed), original)
        self.assertEqual(controller.decode_log_json(framed), original)

    def test_prefix_splitting_a_float_and_a_string(self):
        self.assertEqual(controller.decode_log_json('{"x":3.12[1,0]<stdout>:345,"s":"a[1,0]<stdout>:bc"}'),
                         dict(x=3.12345, s='abc'))


if __name__ == '__main__':
    unittest.main()
