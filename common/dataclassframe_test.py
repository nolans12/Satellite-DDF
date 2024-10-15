import dataclasses
import unittest

from common import dataclassframe


@dataclasses.dataclass
class TestClass:
    a: int
    b: str


class TestDataClassFrame(unittest.TestCase):
    def test_insert_data(self):
        frame = dataclassframe.DataClassFrame(clz=TestClass)

        frame.append(TestClass(1, 'a'))
        frame.append(TestClass(2, 'b'))

        self.assertEqual(len(frame), 2)
        self.assertEqual(frame['a'].tolist(), [1, 2])
        self.assertEqual(frame['b'].tolist(), ['a', 'b'])

    def test_to_dataclasses(self):
        data = [
            TestClass(1, 'a'),
            TestClass(2, 'b'),
        ]
        frame = dataclassframe.DataClassFrame(data, clz=TestClass)

        data = frame.to_dataclasses()

        self.assertEqual(len(data), 2)
        self.assertEqual(data[0], TestClass(1, 'a'))
        self.assertEqual(data[1], TestClass(2, 'b'))

    def test_operated_on(self):
        data = [
            TestClass(1, 'a'),
            TestClass(2, 'b'),
        ]
        frame = dataclassframe.DataClassFrame(data, clz=TestClass)

        # Operating on the frame will return a regular DataFrame
        subset = frame[frame['a'] > 1]

        self.assertEqual(subset['a'].tolist(), [2])
        # But our original frame can accept the new frame as an arg
        self.assertEqual(frame.to_dataclasses(subset), [TestClass(2, 'b')])


if __name__ == '__main__':
    unittest.main()
