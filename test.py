from utils.json_utils import get_by_id, NumpyEncoder, map_from_id, map_to_id, get_by_id_map
from init import init
from app import set_output

models, data, index_to_id, id_to_index = init()
model = models['lda_av']
set_output(model,1)

docs = get_by_id_map(data, ['WMMF'], id_to_index)

print(model.named_steps['embedder'].lda)

parsed = model.named_steps['json'].transform(docs)
vector = model.named_steps['embedder'].transform(parsed)
nn = model.named_steps['tree'].transform(vector)
result = model.named_steps['output'].predict(nn)
output = [{"score": d, "quick_code": index_to_id[index]}
              for d, index in zip(result[0], result[1])]

print(output)
print(parsed)
print(vector)
print(nn)
print(result)

while True:
    parsed = model.named_steps['json'].transform(docs)
    vector = model.named_steps['embedder'].transform(parsed)
    nn = model.named_steps['tree'].transform(vector)
    result = model.named_steps['output'].predict(nn)
    output = [{"score": d, "quick_code": index_to_id[index]}
                for d, index in zip(result[0], result[1])]
    if output[0]['quick_code'] != 'WMMF':
        print(output)
        print(parsed)
        print(vector)
        print(nn)
        print(result)
        break


