import ray
import pyarrow as pa

ray.init(address="auto")

def cast(t: pa.Table) -> pa.Table:
    schema = t.schema
    field_idx = schema.get_field_index("review_body")
    field = schema.field(field_idx)
    new_field = field.with_type(pa.large_string())
    new_schema = schema.set(field_idx, new_field)
    return t.cast(new_schema)

ds = ray.data.read_parquet("s3://dsoaws/parquet/", _block_udf=cast)

ds.groupby("product_category").count().show()
