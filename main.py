from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
#from authentication.auth import router as auth_router
#from authentication.forget_password import router as forget_pass_router
#from users.profile import router as profile_router
#from users.change_password import router as change_pass_router
#from admin.roles import router as roles_router
#from admin.sessions import router as sessions_router
from users.main_function.prepare_image.image import router as prepare_image
from users.main_function.OCR.OCR import router as OCR
#from users.main_function.metadata.metadata import router as metadata_router 
from users.main_function.QA.AI_QA import router as AI_QA_router
from models.database import Base, engine
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)
Base.metadata.create_all(bind=engine)
#app.include_router(auth_router)
#User
#app.include_router(profile_router)
#app.include_router(forget_pass_router)
#app.include_router(change_pass_router)
#app.include_router(roles_router)
#app.include_router(sessions_router)

# Main function
app.include_router(prepare_image)
app.include_router(OCR)
#app.include_router(metadata_router)
app.include_router(AI_QA_router)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)