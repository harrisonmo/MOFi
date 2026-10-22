"""Recovery preserves attempts and the scientific workflow."""
import unittest
from rsi.q2b_recovery import Recovery, remaining, LIMITS
from rsi.q2b_v21.run import Pilot


class RecoveryTests(unittest.TestCase):
    def test_preserves_failed_attempts(self):
        schedule = [{'id': str(i), 'arm': 'fixed'} for i in range(96)]
        attempts = [dict(r, passed=False) for r in schedule[:45]]
        self.assertEqual(remaining(schedule, attempts), schedule[45:])
        self.assertEqual(len(remaining(schedule, attempts)), 51)

    def test_rejects_nonprefix_and_duplicates(self):
        rows = [{'id': 'a', 'arm': 'fixed'}, {'id': 'b', 'arm': 'fixed'}]
        for schedule, done in [(rows, rows[1:]), (rows + rows, []), (rows, rows[:1] * 2)]:
            with self.assertRaises(ValueError):
                remaining(schedule, done)

    def test_workflow_and_remaining_budget(self):
        self.assertIs(Recovery.episode, Pilot.episode)
        self.assertEqual(LIMITS['total_model_calls'] + 93, 197)
        self.assertEqual(LIMITS['research_calls'], 51 * 2)
        self.assertEqual(LIMITS['generation_and_debug_calls'], 0)


if __name__ == '__main__':
    unittest.main()
