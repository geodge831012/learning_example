# coding:utf-8


def generate_news_id(ipublish, _id, db_source_id, table_index=1):
    news_id = '%d_%d_%d_%d' % (ipublish, _id, db_source_id, table_index)
    return news_id
