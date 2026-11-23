"""Task 5 checks: aggregation, policy contract, reproducibility, and checkpoints."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import io

import numpy as np

import poker_env as pe
from evaluate_policies import evaluate_policy, load_checkpoint, summarize
import evaluate_policies
from q_learning import QLearningAgent
from plot_evaluation import comparisons, learning_curves, checkpoint_curves, greedy_figures


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.hands = np.array([pe.parse_hand('Ah Ad Ac As 2h'),
                               pe.parse_hand('2h 4s 6d 8c Th')])
        self.classes = np.array([7, 0])

    def test_keep_all_known_rewards_and_missing_classes(self):
        r = evaluate_policy(lambda h: 31, self.hands, self.classes, seed=0, draws=3)
        np.testing.assert_array_equal(r['per_hand'], [25, 0])
        self.assertEqual(r['eval_mean'], 12.5)
        self.assertEqual(r['n_outcomes'], 6)
        self.assertEqual(r['by_class']['royal_flush']['n_hands'], 0)
        self.assertIsNone(r['by_class']['royal_flush']['mean'])
        self.assertIsNone(r['by_class']['four_of_a_kind']['ci95'])

    def test_clustered_interval_not_independent_repeated_draws(self):
        one = evaluate_policy(lambda h:31, self.hands, self.classes, seed=0, draws=1)
        many = evaluate_policy(lambda h:31, self.hands, self.classes, seed=0, draws=20)
        self.assertEqual(one['eval_ci95'], many['eval_ci95'])
        self.assertEqual(summarize([1, 3])['mean'], 2)

    def test_reproducible_draws_and_invalid_actions(self):
        def run():
            return evaluate_policy(lambda h:0, self.hands, self.classes, seed=8, draws=10)
        np.testing.assert_array_equal(run()['per_hand'], run()['per_hand'])
        with self.assertRaises(ValueError):
            evaluate_policy(lambda h:32, self.hands, self.classes, seed=1)
        with self.assertRaises(ValueError):
            evaluate_policy(lambda h:1.5, self.hands, self.classes, seed=1)

    def test_checkpoint_evaluation_leaves_agent_unchanged(self):
        agent = QLearningAgent(seed=3)
        # Exercise canonical mapping with a nontrivial winning action.
        for hand in self.hands:
            state, perm = agent.encoder(hand)
            agent.Q[state, 5] = 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.npz'
            agent.save(path)
            loaded = load_checkpoint(path)
            self.assertEqual(loaded(self.hands[0]), agent(self.hands[0]))
            before = loaded.Q.copy(), loaded.N.copy(), json.dumps(loaded.rng.bit_generator.state)
            evaluate_policy(loaded,self.hands,self.classes,seed=9,draws=2)
            np.testing.assert_array_equal(before[0],loaded.Q)
            np.testing.assert_array_equal(before[1],loaded.N)
            self.assertEqual(before[2],json.dumps(loaded.rng.bit_generator.state))
            self.assertEqual(loaded.episode,0)

    def test_ranks_checkpoint_keeps_original_card_positions(self):
        from abstraction import get_abstraction
        abstraction = get_abstraction('ranks')
        agent = QLearningAgent(n_states=abstraction.n_states, state_encoding='ranks',
                               encoder=abstraction.state_encoder())
        hand = pe.parse_hand('Ks 2h Kd 7c Ah')
        state, perm = agent.encoder(hand)
        mask = pe.original_mask_to_canonical(pe.mask_from_keep([0,2]), perm)
        agent.Q[state, mask] = 5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'ranks.npz'
            agent.save(path)
            loaded = load_checkpoint(path)
            self.assertEqual(loaded(hand), pe.mask_from_keep([0,2]))
            reordered = hand[[4,2,3,0,1]]
            self.assertEqual(loaded(reordered), pe.mask_from_keep([1,3]))
            result = evaluate_policy(loaded, np.array([hand,reordered]), [1,1], seed=0, draws=2)
            self.assertEqual(result['n_outcomes'],4)

    def test_reject_bad_hands_and_summary(self):
        for hand in [[0,0,1,2,3], [0,1,2,3,52], [0.,1.,2.,3.,4.]]:
            with self.assertRaises(ValueError):
                evaluate_policy(lambda h:31, [hand], [0], seed=0)
        with self.assertRaises(ValueError):
            summarize([np.nan])

    def test_exact_greedy_checkpoint_uses_existing_evaluator(self):
        from evaluate_checkpoints import score_checkpoint
        agent = QLearningAgent(seed=2)
        for hand in self.hands:
            state, _ = agent.encoder(hand)
            agent.Q[state,31] = 1
        split = dict(state_id=pe.state_id_batch(self.hands),uci_class=self.classes)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'default/seed2/checkpoint.npz'
            agent.save(path)
            result=score_checkpoint(path,split)
            self.assertEqual(result['eval_mean'],12.5)
            self.assertEqual(result['seed'],2)
            self.assertEqual(result['configuration'],'default')

    def test_final_greedy_figures_require_complete_results(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            source=output/'scores.json'
            source.write_text(json.dumps(dict(complete=False)))
            greedy_figures(source,output)
            self.assertFalse((output/'task2_learning_curves.png').exists())
            records=[]
            for seed in range(5):
                for episode in [100,200]:
                    records.append(dict(configuration='default',seed=seed,episodes=episode,
                        eval_mean=seed+episode/100,visited_action_fraction=.5,by_class=dict(nothing=.2)))
            source.write_text(json.dumps(dict(complete=True,target_episodes=200,records=records)))
            greedy_figures(source,output)
            self.assertTrue((output/'task2_learning_curves.png').exists())
            self.assertIn('training_seed_sd',(output/'task2_final_comparison.csv').read_text())

    def test_saved_sample_curves(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            paths=[]
            for seed, scores in [(0,[.2,.4]),(1,[.4,.6])]:
                path=folder/f'{seed}.json'
                path.write_text(json.dumps(dict(config=dict(reward='job_9_6',abstraction='dummy'),
                    seed=seed,episodes=20,eval_mean=scores[-1],runtime_s=1,
                    curve=[dict(episodes=n,eval_mean=v) for n,v in zip([10,20],scores)])))
                paths.append(path)
            learning_curves(paths,folder)
            self.assertTrue((folder/'learning_curves.png').exists())
            self.assertIn('training_seed_sd',(folder/'learning_curves.csv').read_text())

    def test_comparison_clusters_seeds_by_hand_and_checkpoint_curves(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            split=Path(__file__).parent/'data/splits/eval.npz'
            np.savez(folder/'per_hand.npz',indices=[0,1],a=[1.,3.],b=[3.,5.])
            records=[]
            for seed,key,mean in [(0,'a',2.),(1,'b',4.)]:
                records.append(dict(policy='dummy_checkpoint',values_key=key,eval_seed=seed,
                    eval_mean=mean,runtime_s=1.,decision_runtime_s=.1,
                    checkpoint='dummy.npz',config=dict(seed=4,state_encoding='canonical'),
                    seed=4,episodes=10))
            source=folder/'evaluation.json'
            source.write_text(json.dumps(dict(split_sha256=hashlib.sha256(split.read_bytes()).hexdigest(),
                                             records=records)))
            rows=comparisons(source,folder)
            self.assertEqual(rows[0]['mean'],3.)
            # Mean per hand is [2,4], so SE is 1 (not four independent observations).
            self.assertAlmostEqual(rows[0]['ci95_high'],4.96)
            checkpoint_curves(source,folder)
            self.assertTrue((folder/'checkpoint_learning_curves.png').exists())

    def test_cli_baselines_and_checkpoint_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            checkpoint=folder/'checkpoint_0.npz'
            QLearningAgent(seed=2).save(checkpoint)
            out=folder/'results'
            argv=['evaluate_policies.py','--output',str(out),'--hands','10','--draws','2',
                  '--seeds','0','1','--checkpoints',str(checkpoint)]
            with patch('sys.argv',argv), redirect_stdout(io.StringIO()):
                evaluate_policies.main()
            data=json.loads((out/'evaluation.json').read_text())
            self.assertEqual(len(data['records']),12)
            self.assertEqual(data['method'],'monte_carlo')
            self.assertEqual(data['records'][-1]['episodes'],0)
            self.assertEqual(data['records'][-1]['seed'],2)
            with np.load(out/'per_hand.npz') as values:
                self.assertEqual(len(values['indices']),10)
                self.assertEqual(len(values.files),13)
            self.assertEqual(len(comparisons(out/'evaluation.json',out)),6)


if __name__=='__main__':
    unittest.main()
