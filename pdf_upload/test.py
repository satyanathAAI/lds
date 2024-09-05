from api_utils.mongo_utils import check_cache


result,_=check_cache(videoname="doctrine-and-covenants-eng.pdf",collection_name="pdf")
print(result.keys())