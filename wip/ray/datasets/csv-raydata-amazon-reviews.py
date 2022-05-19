import ray
from pyarrow.csv import ParseOptions

class InvalidRowHandler:
    def __init__(self, result):
        self.result = result
        self.rows = []

    def __call__(self, row):
        self.rows.append(row)
        return self.result

    def __eq__(self, other):
        return (isinstance(other, InvalidRowHandler) and
                other.result == self.result)

    def __ne__(self, other):
        return (not isinstance(other, InvalidRowHandler) or
                other.result != self.result)


skip_handler = InvalidRowHandler('skip')

df = ray.data.read_csv(paths='s3://dsoaws/amazon_reviews_us_Digital_Software_v1_00.tsv',
                       parse_options=ParseOptions(delimiter='\t')
#                                                  invalid_row_handler=skip_handler))


print(df)
print('blah')
