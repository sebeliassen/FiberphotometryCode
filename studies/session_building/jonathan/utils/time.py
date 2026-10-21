def parse_session_ts(ts_str):
    from datetime import datetime
    return datetime.strptime(ts_str, "%Y-%m-%dT%H_%M_%S")

def parse_csv_ts(ts_str):
    from datetime import datetime
    return datetime.strptime(ts_str, "%Y-%m-%dT%H.%M.%S")
