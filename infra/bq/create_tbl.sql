CREATE TABLE <project>.default_dataset.span (
  name STRING,
  trace_id STRING,
  span_id STRING,
  parent_id STRING,
  span_kind STRING,
  start_time TIMESTAMP,
  end_time TIMESTAMP,
  attributes JSON,
  events JSON ,
  status_code STRING,
  status_message STRING
);

DROP TABLE IF EXISTS `<project>.default_dataset.span`;