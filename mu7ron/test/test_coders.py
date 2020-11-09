import unittest

from mu7ron import analyze
from mu7ron import coders
from mu7ron.test import _test_variables
from mu7ron.test import _test_parameters


class TestCoders(unittest.TestCase):
    def test_categorize(self):
        """
        This method will test the ability of the translate module to convert from a midi sequence with
        midi.events objects to a sequence of base 10 integers. This is a serialization of the data in
        that we achieve a one dimensional categorical sequence as a result. The categorical sequence is 
        then converted back and we test the correct values are achieved at each stage.
        """
        print("\ntesting mu7ron.translate.categorize_input...")
        for i in range(4):
            p = eval(f"_test_parameters.p{i}()")
            for j in range(7):
                v = eval(f"_test_variables.v{i}{j}()")

                result = coders.categorize_input(
                    v["uncategorized"],
                    p["q"],
                    p["n_time"],
                    p["off_mode"],
                    p["time_encoder"],
                    p["ekwa"],
                )

                self.assertTrue(bool(len(result)))
                self.assertTrue(max(result) < p["n_vocab"])
                self.assertTrue(min(result) >= 0)
                self.assertEqual(v["categorized"], result)

                result = coders.decategorize_output(
                    result,
                    p["q"],
                    p["n_time"],
                    p["off_mode"],
                    p["time_decoder"],
                    p["dkwa"],
                )

                self.assertTrue(
                    analyze.is_equal_midi_sequences(v["decategorized"], result)
                )

    def test_decategorize(self):
        print("\ntesting mu7ron.translate.decategorize_output...")
        for i in range(4):
            p = eval(f"_test_parameters.p{i}()")
            for j in range(7):
                v = eval(f"_test_variables.v{i}{j}()")

                result = coders.decategorize_output(
                    v["categorized"],
                    p["q"],
                    p["n_time"],
                    p["off_mode"],
                    p["time_decoder"],
                    p["dkwa"],
                )
                self.assertTrue(
                    analyze.is_equal_midi_sequences(v["decategorized"], result)
                )


if __name__ == "__main__":
    unittest.main()
