"""Global batch, fixed decay, and exact original packed update data contracts."""
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/cloud'))
from run_lion_ttt_full_batch32 import arguments
from lion_ttt_four_gpu_train import RebatchReader


class Full32Tests(unittest.TestCase):
    def test_global_batch_and_fixed_wsd(self):
        for name,target,decay in (('1xc',78216,7546),('2xc',156432,15091)):
            args=arguments(name,'lion_muon',target,decay,False)
            batch=int(args[args.index('--batch-size')+1]);acc=int(args[args.index('--acc-steps')+1])
            divisor=math.gcd(acc,4)
            self.assertEqual((batch//(4//divisor),acc//divisor),(32,1))
            self.assertEqual(batch*acc,128)
            self.assertEqual(int(float(args[args.index('--wsd-fract-decay')+1])*target),decay)
            self.assertIn('--no-local-save',args)
            self.assertEqual(args[args.index('--latest-ckpt-interval')+1],'0')
            self.assertEqual(args[args.index('--permanent-ckpt-interval')+1],'0')
    def test_rebatch_preserves_update_examples_and_cursor(self):
        import torch
        class Source:
            def __init__(self,rank):self.rank=rank;self.step=0
            def set_step(self,step):self.step=step
            def sample_batch(self):
                source,shard=divmod(self.rank,2)
                rows=torch.arange(self.step*16+shard*8,self.step*16+shard*8+8)+source*100000
                self.step+=1
                return rows,rows+1
        seen=[]
        for update in (0,1,75456,156431):
            current=[]
            for rank in range(4):
                expected_source=Source(rank);expected_source.set_step(update*4)
                expected=torch.cat([expected_source.sample_batch()[0] for _ in range(4)])
                reader=RebatchReader(Source(rank));reader.set_step(update)
                x,y=reader.sample_batch()
                self.assertTrue(torch.equal(x,expected));self.assertTrue(torch.equal(y,x+1))
                self.assertEqual(reader.source.step,(update+1)*4)
                current.extend(x.tolist())
            self.assertEqual(len(set(current)),128)
            if update<2:seen.append(set(current))
        self.assertFalse(seen[0]&seen[1])


if __name__=='__main__':unittest.main()
