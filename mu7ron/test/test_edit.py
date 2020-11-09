import unittest

import midi

from mu7ron import analyze
from mu7ron import edit
from mu7ron.test import _test_variables
from mu7ron.test import _test_parameters


class TestEdit(unittest.TestCase):
    def test_filter_ptrn_of_evnt_typs(self):
        """
        This method will test mu7ron.edit.filter_ptrn_of_evnt_typs
        """
        print("\ntesting mu7ron.edit.filter_ptrn_of_evnt_typs...")
        for i in range(4):
            p = eval(f"_test_parameters.p{i}()")
            for j in range(7):
                v = eval(f"_test_variables.v{i}{j}()")
                result = edit.filter_ptrn_of_evnt_typs(v["opened"], p["typs_2_keep"])
                self.assertTrue(
                    analyze.is_equal_midi_sequences(v["evnt_filtered"], result)
                )

    def test_filter_ptrn_of_insts(self):
        """
        This method will test mu7ron.edit.filter_ptrn_of_insts
        """
        print("\ntesting mu7ron.edit.filter_ptrn_of_insts...")
        for i in range(4):
            p = eval(f"_test_parameters.p{i}()")
            for j in range(7):
                v = eval(f"_test_variables.v{i}{j}()")
                result = edit.filter_ptrn_of_insts(
                    v["evnt_filtered"],
                    condition=lambda x: x.data[0] == 47 or x.data[0] > 112,
                )
                self.assertTrue(
                    analyze.is_equal_midi_sequences(v["inst_filtered"], result)
                )

    def test_filter_ptrn_of_insts(self):
        """
        This method will test mu7ron.edit.filter_ptrn_of_percussion
        """
        print("\ntesting mu7ron.edit.filter_ptrn_of_percussion...")
        for i in range(4):
            p = eval(f"_test_parameters.p{i}()")
            for j in range(7):
                v = eval(f"_test_variables.v{i}{j}()")
                result = edit.filter_ptrn_of_percussion(v["inst_filtered"])
                self.assertTrue(
                    analyze.is_equal_midi_sequences(v["perc_filtered"], result)
                )

    def test_consolidate_trcks(self):
        """
        This method will test mu7ron.edit.consolidate_trcks
        """
        print("\ntesting mu7ron.edit.consolidate_trcks...")
        for i in range(4):
            p = eval(f"_test_parameters.p{i}()")
            for j in range(7):
                v = eval(f"_test_variables.v{i}{j}()")
                result = edit.consolidate_trcks(v["perc_filtered"])
                self.assertTrue(
                    analyze.is_equal_midi_sequences(v["consolidated"], result)
                )

    def test_normalize_resolution(self):
        """
        This method will test mu7ron.edit.normalize_resolution
        """
        print("\ntesting mu7ron.edit.normalize_resolution...")
        for i in range(4):
            p = eval(f"_test_parameters.p{i}()")
            for j in range(7):
                v = eval(f"_test_variables.v{i}{j}()")
                result = edit.normalize_resolution(v["consolidated"], res=480)
                self.assertTrue(result.resolution == 480)
                self.assertTrue(
                    analyze.is_equal_midi_sequences(v["normalized"], result)
                )

    def test_quantize_typ_attr(self):
        """
        This method will test mu7ron.edit.quantize_typ_attr
        """
        print("\ntesting mu7ron.edit.quantize_typ_attr...")
        for i in range(4):
            p = eval(f"_test_parameters.p{i}()")
            for j in range(7):
                v = eval(f"_test_variables.v{i}{j}()")
                result = edit.quantize_typ_attr(
                    v["normalized"],
                    p["q"],
                    (midi.NoteOnEvent, midi.NoteOffEvent),
                    lambda x: x.data[1],
                )
                self.assertTrue(analyze.is_equal_midi_sequences(v["quantized"], result))

    def test_dedupe(self):
        """
        This method will test the ability of the edit module...
        """
        print("\ntesting mu7ron.edit.dedupe...")
        for i in range(4):
            p = eval(f"_test_parameters.p{i}()")
            for j in range(7):
                v = eval(f"_test_variables.v{i}{j}()")
                result = edit.dedupe(v["quantized"])
                self.assertTrue(analyze.is_equal_midi_sequences(v["deduped"], result))

        test0 = midi.Pattern()
        test0.append(
            midi.Track(
                [
                    midi.NoteOnEvent(tick=0, data=[1, 0]),
                    midi.NoteOnEvent(tick=0, data=[1, 0]),
                    midi.NoteOnEvent(tick=1, data=[1, 0]),
                ]
            )
        )

        target0 = midi.Pattern()
        target0.append(
            midi.Track(
                [
                    midi.NoteOnEvent(tick=0, data=[1, 0]),
                    midi.NoteOnEvent(tick=1, data=[1, 0]),
                ]
            )
        )

        test1 = midi.Pattern()
        test1.append(
            midi.Track(
                [
                    midi.NoteOnEvent(tick=0, data=[1, 0]),
                    midi.NoteOnEvent(tick=1, data=[1, 0]),
                    midi.NoteOnEvent(tick=0, data=[1, 0]),
                ]
            )
        )

        target1 = target0

        test2 = midi.Pattern()
        test2.append(
            midi.Track(
                [
                    midi.NoteOnEvent(tick=0, data=[1, 0]),
                    midi.NoteOffEvent(tick=0, data=[1, 0]),
                    midi.NoteOnEvent(tick=0, data=[1, 1]),
                ]
            )
        )

        target2 = test2

        test3 = midi.Pattern()
        test3.append(
            midi.Track(
                [
                    midi.NoteOnEvent(tick=0, data=[1, 0]),
                    midi.NoteOffEvent(tick=1, data=[1, 0]),
                    midi.NoteOffEvent(tick=0, data=[1, 0]),
                    midi.NoteOnEvent(tick=0, data=[1, 1]),
                ]
            )
        )

        target3 = midi.Pattern()
        target3.append(
            midi.Track(
                [
                    midi.NoteOnEvent(tick=0, data=[1, 0]),
                    midi.NoteOffEvent(tick=1, data=[1, 0]),
                    midi.NoteOnEvent(tick=0, data=[1, 1]),
                ]
            )
        )

        test4 = midi.Pattern()
        test4.append(
            midi.Track(
                [
                    midi.NoteOnEvent(tick=0, data=[1, 0], channel=0),
                    midi.NoteOnEvent(tick=0, data=[1, 0], channel=1),
                    midi.NoteOnEvent(tick=0, data=[1, 0], channel=9),
                    midi.NoteOnEvent(tick=0, data=[1, 0], channel=9),
                ]
            )
        )

        target4 = midi.Pattern()
        target4.append(
            midi.Track(
                [
                    midi.NoteOnEvent(tick=0, data=[1, 0], channel=0),
                    midi.NoteOnEvent(tick=0, data=[1, 0], channel=9),
                ]
            )
        )

        test5 = midi.Pattern()
        test5.append(
            midi.Track(
                [
                    midi.NoteOffEvent(tick=0, data=[1, 0], channel=1),
                    midi.NoteOffEvent(tick=0, data=[1, 0], channel=1),
                    midi.NoteOffEvent(tick=0, data=[1, 0], channel=9),
                    midi.NoteOffEvent(tick=0, data=[1, 1], channel=9),
                ]
            )
        )

        target5 = midi.Pattern()
        target5.append(
            midi.Track(
                [
                    midi.NoteOffEvent(tick=0, data=[1, 0], channel=1),
                    midi.NoteOffEvent(tick=0, data=[1, 0], channel=9),
                    midi.NoteOffEvent(tick=0, data=[1, 1], channel=9),
                ]
            )
        )

        result0 = edit.dedupe(test0)
        self.assertTrue(analyze.is_equal_midi_sequences(target0, result0))
        result1 = edit.dedupe(test1)
        self.assertTrue(analyze.is_equal_midi_sequences(target1, result1))
        result2 = edit.dedupe(test2)
        self.assertTrue(analyze.is_equal_midi_sequences(target2, result2))
        result3 = edit.dedupe(test3)
        self.assertTrue(analyze.is_equal_midi_sequences(target3, result3))
        result4 = edit.dedupe(test4)
        self.assertTrue(analyze.is_equal_midi_sequences(target4, result4))
        result5 = edit.dedupe(test5)

        self.assertTrue(analyze.is_equal_midi_sequences(target5, result5))

    def test_replace_evnt_typ(self):
        """
        This method will test mu7ron.edit.replace_evnt_typ
        """

        def copy_func(old_evnt, new_typ):
            new_evnt = new_typ(
                tick=old_evnt.tick,
                data=[old_evnt.data[0], 0],
                channel=old_evnt.channel,
            )
            return new_evnt

        print("\ntesting mu7ron.edit.replace_evnt_typ...")
        for i in range(4):
            p = eval(f"_test_parameters.p{i}()")
            if not p["off_mode"]:
                for j in range(7):
                    v = eval(f"_test_variables.v{i}{j}()")
                    result = edit.replace_evnt_typ(
                        v["deduped"],
                        midi.NoteOffEvent,
                        midi.NoteOnEvent,
                        copy_func=copy_func,
                    )
                    self.assertTrue(
                        analyze.is_equal_midi_sequences(v["nonoteoffs"], result)
                    )

    def test_time_split(self):
        """
        This method will test mu7ron.edit.time_split
        """
        print("\ntesting mu7ron.edit.time_split...")
        trck_tests, ptrn_tests = _test_variables.v42()
        targets = [
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]],
            [[0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
            [[1, 0], [1, 0], [1, 0, 0, 0], [1], [1]],
            [[150], [150], [150, 0], [150], [150, 0], [150, 0]],
            [[1], [1], [1, 0, 0, 0], [1], [1, 0, 0]],
            [[1], [4], [999, 0, 0], [678, 0], [123], [1], [1, 0], [1]],
            [
                [0, 0, 0, 0],
                [5, 0, 0, 0, 0, 0, 0, 0],
                [15, 0, 0, 0],
                [20, 0, 0, 0, 0, 0],
                [123, 0],
                [876, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [5, 0, 0, 0, 0, 0, 0, 0],
                [15, 0, 0, 0],
                [20, 0, 0, 0, 0, 0],
                [123, 0],
                [876, 0, 0, 0],
            ],
            [],
            [[]],
        ]
        trck_targets = targets
        ptrn_targets = [*targets[:-2], targets[-1], targets[-1]]
        for t_tst, p_tst, t_trgt, p_trgt in zip(
            trck_tests, ptrn_tests, trck_targets, ptrn_targets
        ):
            self.assertEqual(eval(str(edit.time_split(t_tst))), t_trgt)
            self.assertEqual(eval(str(edit.time_split(p_tst))), p_trgt)


if __name__ == "__main__":
    unittest.main()
