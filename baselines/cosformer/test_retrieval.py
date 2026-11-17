import unittest
import numpy as np
from retrieval import select_representatives, merge_surviving_buffer


class RetrievalTests(unittest.TestCase):
    def test_train_only_and_alignment_dimensions(self):
        with self.assertRaisesRegex(ValueError, 'Non-training'):
            select_representatives([('test', 0, np.eye(2))], {0: [1, 0]}, {'train'})
        with self.assertRaisesRegex(ValueError, 'dimensions differ'):
            select_representatives([('a', 0, np.eye(2))], {0: [1, 0, 0]}, {'a'})

    def test_selection_deterministic_and_per_class(self):
        bags = [('a', 0, [[1, 0], [.9, .1]]), ('b', 0, [[0, 1], [.1, .9]]),
                ('c', 1, [[0, 1], [.2, .8]])]
        text = {0: [1, 0], 1: [0, 1]}
        selected, audit = select_representatives(bags, text, {'a', 'b', 'c'}, 1)
        self.assertEqual(selected, ['a', 'c'])
        self.assertEqual(audit[0]['cluster_size'], 2)
        self.assertEqual(select_representatives(list(reversed(bags)), text,
                                               {'a', 'b', 'c'}, 1)[0], selected)

    def test_memory_only_merges_survivors(self):
        items = [{'slide_id': str(i)} for i in range(8)]
        kept = merge_surviving_buffer(items[:3], items[3:], 4, np.random.default_rng(1))
        self.assertEqual(len(kept), 4)
        next_kept = merge_surviving_buffer(kept, [{'slide_id': 'new'}], 4,
                                          np.random.default_rng(2))
        self.assertTrue({r['slide_id'] for r in next_kept} <=
                        {r['slide_id'] for r in kept} | {'new'})


if __name__ == '__main__':
    unittest.main()
