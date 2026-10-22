from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.security import OAuth2PasswordRequestForm

from app.api.deps import CurrentUser, DbSession
from app.api.limits import Limiter, enforce_login_email, limit_default, limit_login, limit_register
from app.core.security import create_access_token
from app.schemas.auth import Token
from app.schemas.user import UserCreate, UserRead
from app.services import users as user_service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[limit_register],
)
async def register(payload: UserCreate, db: DbSession):
    return await user_service.register_user(db, payload.email, payload.password)


@router.post("/login", response_model=Token, dependencies=[limit_login])
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: DbSession,
    limiter: Limiter,
    response: Response,
):
    # OAuth2 form fields are named "username"/"password"; username holds the email.
    await enforce_login_email(limiter, user_service.normalize_email(form.username), response)
    user = await user_service.authenticate(db, form.username, form.password)
    if user is None:
        # One message for both "no such email" and "wrong password".
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Token(access_token=create_access_token(user.id))


@router.get("/me", response_model=UserRead, dependencies=[limit_default])
async def me(current_user: CurrentUser):
    return current_user
