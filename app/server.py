from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
import os

from fastai import *
from fastai.vision import *

# export_file_url = 'https://www.dropbox.com/s/v6cuuvddq73d1e0/export.pkl?raw=1'
# export_file_url = 'https://www.dropbox.com/s/6bgq8t6yextloqp/export.pkl?raw=1'
# export_file_name = 'export.pkl'

classes = ['Melanocytic Nevi', 'Melanoma', 'Benign Keratosis', 'Basal Cell Carcinoma', 'Actinic Keratoses', 'Vascular Lesions', 'Dermatofibroma']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

# async def download_file(url, dest):
#     if dest.exists(): return
#     async with aiohttp.ClientSession() as session:
#         async with session.get(url) as response:
#             data = await response.read()
#             with open(dest, 'wb') as f: f.write(data)

class CustomImageItemList(ImageList):
    def custom_label(self,df, **kwargs)->'LabelList':
        """Custom Labels from path"""
        file_names=np.vectorize(lambda files: str(files).split('/')[-1][:-4])
        get_labels=lambda x: df.loc[x,'lesion']
        #self.items is an np array of PosixPath objects with each image path
        labels= get_labels(file_names(self.items))
        y = CategoryList(items=labels)
        res = self._label_list(x=self,y=y)
        return res

async def setup_learner():
    # await download_file(export_file_url, path/export_file_name)
    try:
        learn = load_learner('app/models')
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)
    probabilities = prediction[2]
    top_3_probabilities = torch.sort(probabilities,descending=True)[0][:3]
    top_3_names = torch.sort(probabilities,descending=True)[1][:3]
    # result = {classes[top_3_names[0].item()]:str("%.2f"%(100*top_3_probabilities[0]).item()),classes[top_3_names[1].item()]:str("%.2f"%(100*top_3_probabilities[1].item())),classes[top_3_names[2].item()]:str("%.2f"%(100*top_3_probabilities[2].item()))}
    result = [{"name":classes[top_3_names[0].item()],"probability":str("%.2f"%(100*top_3_probabilities[0]).item())},{"name":classes[top_3_names[1].item()],"probability":str("%.2f"%(100*top_3_probabilities[1]).item())},{"name":classes[top_3_names[2].item()],"probability":str("%.2f"%(100*top_3_probabilities[2]).item())}]

    return JSONResponse({"result":result})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
