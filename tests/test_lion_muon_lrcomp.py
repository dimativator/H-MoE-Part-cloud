import unittest
import torch
from optim.lion_muon_ablation import LionMuonAblation, LionMuonLRCompensation


class LRCompensationTest(unittest.TestCase):
    def make(self, gamma, ratio=.01):
        p=torch.nn.Parameter(torch.zeros(5, 3));a=torch.nn.Parameter(torch.zeros(3))
        o=LionMuonLRCompensation([p],variant='probability_down',total_steps=101,
            compensation_gamma=gamma,lr=.001,lion_lr=.001*ratio,adamw_params=[a])
        return o,p,a

    def test_probability_and_lr_endpoints(self):
        for gamma in (0.,.5,1.):
            for ratio in (.01,.02):
                o,_,_=self.make(gamma,ratio)
                for step in range(101):
                    o._step_count=step;o._is_muon_step();p=.35-.3*step/100
                    for base in (.001,.0003,0.):
                        o.param_groups[0]['lr']=base
                        m,l=o._branch_lrs(o.param_groups[0])
                        self.assertAlmostEqual(m,base*(.2/p)**gamma)
                        self.assertAlmostEqual(l,base*ratio*(.8/(1-p))**gamma)
                        if gamma==1:
                            self.assertAlmostEqual(p*m,.2*base)
                            self.assertAlmostEqual((1-p)*l,.8*base*ratio)
                        elif gamma==.5:
                            self.assertAlmostEqual(p*m*m,.2*base*base)
                            self.assertAlmostEqual((1-p)*l*l,.8*(base*ratio)**2)

    def test_gamma_zero_matches_baseline_and_auxiliary_unchanged(self):
        torch.manual_seed(1)
        baseline_p=torch.nn.Parameter(torch.zeros(5,3));baseline_a=torch.nn.Parameter(torch.zeros(3))
        baseline=LionMuonAblation([baseline_p],variant='probability_down',total_steps=101,
                                lr=.001,lion_lr=.00001,adamw_params=[baseline_a])
        opts=[self.make(g) for g in (0.,.5,1.)]
        for step in range(30):
            g=torch.randn_like(baseline_p);ag=torch.randn_like(baseline_a)
            baseline_p.grad=g.clone();baseline_a.grad=ag.clone();baseline.step()
            for opt,p,a in opts:
                p.grad=g.clone();a.grad=ag.clone();opt.step()
                self.assertEqual(opt.last_muon,baseline.last_muon)
                torch.testing.assert_close(a,baseline_a,rtol=0,atol=0)
                torch.testing.assert_close(opt.state[p]['exp_avg'],baseline.state[baseline_p]['exp_avg'],rtol=0,atol=0)
                if opt.compensation_gamma==0:torch.testing.assert_close(p,baseline_p,rtol=0,atol=0)
                self.assertNotIn('neuron_variance',opt.state[p])
                self.assertEqual(opt.param_groups[0]['lr'],.001)

    def test_launcher(self):
        from scripts.cloud.run_lionmuon_lrcomp import TRIALS,TRIAL_CONFIG,arguments
        self.assertEqual(len(TRIALS),5)
        for n in TRIALS:
            cmd=arguments(n,'lion_muon',75457,7546,False);get=lambda f:cmd[cmd.index(f)+1]
            g,ratio=TRIAL_CONFIG[n]
            self.assertEqual(float(get('--lion-compensation-gamma')),g)
            self.assertEqual(float(get('--lion-lr-ratio')),ratio)
            self.assertEqual(get('--lion-ablation'),'probability_down')
            self.assertEqual(get('--lion-probability-seed'),'20261117')
            self.assertEqual(get('--latest-ckpt-interval'),'0')
            self.assertEqual(get('--permanent-ckpt-interval'),'0')
            self.assertIn('--no-local-save',cmd)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
