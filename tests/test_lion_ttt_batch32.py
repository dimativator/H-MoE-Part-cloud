"""Guard the distinction between microbatch32 and effective batch32."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/cloud'))
from benchmark_lion_ttt_batch32 import command

class Batch32Tests(unittest.TestCase):
    def test_no_accumulation_for_both_optimizers(self):
        for opt in ('muon_reference','lion_muon'):
            args=command(opt)
            self.assertEqual(args[args.index('--batch-size')+1],'32')
            self.assertEqual(args[args.index('--acc-steps')+1],'1')
            self.assertNotIn('--distributed-backend',args)
            self.assertIn('--no-local-save',args)
            self.assertEqual(args[args.index('--latest-ckpt-interval')+1],'0')
            self.assertEqual(args[args.index('--early-stop-iteration')+1],'240')
if __name__=='__main__':unittest.main()
