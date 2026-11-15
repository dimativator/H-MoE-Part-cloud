"""Global data-order and effective-batch contracts for the single-GPU benchmark."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/cloud'))
from lion_ttt_single_train import prefix_indices
import benchmark_lion_ttt_single as benchmark


class SingleGpuTimingTests(unittest.TestCase):
    def test_microbatch_changes_preserve_the_same_global_examples(self):
        expected=[]
        for source_step in range(4):
            for rank in (0,1):
                expected.extend((rank,i) for i in range(source_step*16,(source_step+1)*16))
        for microbatch in (128,64,32,16,8,4,2,1):
            actual=[]
            for start in range(0,128,microbatch):actual.extend(prefix_indices(start,microbatch))
            self.assertEqual(actual,expected)
            command=benchmark.command('test','lion_muon',microbatch,True)
            self.assertNotIn('--distributed-backend',command)
            self.assertEqual(int(command[command.index('--batch-size')+1])*int(command[command.index('--acc-steps')+1]),128)
            self.assertIn('--no-local-save',command)
    def test_consecutive_updates_do_not_repeat_source_examples(self):
        a=set(prefix_indices(0,128));b=set(prefix_indices(128,128))
        self.assertEqual(len(a),128);self.assertFalse(a&b)
        self.assertEqual(prefix_indices(128,1)[0],(0,64))
    def test_probe_covers_lion_and_muon_steps(self):
        command=benchmark.command('test','lion_muon',64,False)
        self.assertEqual(command[command.index('--early-stop-iteration')+1],'6')
        self.assertEqual(command[command.index('--latest-ckpt-interval')+1],'0')


if __name__=='__main__':unittest.main()
