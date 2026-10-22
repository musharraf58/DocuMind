from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from redis.asyncio import Redis

from app.api.deps import CurrentUser, DbSession
from app.api.limits import Limiter, enforce_login_email, limit_default, limit_login, limit_register
from app.core.client_ip import client_ip
from app.core.redis import get_redis
from app.core.security import create_access_token
from app.schemas.auth import Token
from app.schemas.user import UserCreate, UserRead
from app.services import users as user_service
from app.services.login_guard import LoginGuard

router = APIRouter(prefix="/auth", tags=["auth"])

INCORRECT_LOGIN = "Incorrect email or password"


def get_login_guard(redis: Annotated[Redis, Depends(get_redis)]) -> LoginGuard:
    return LoginGuard(redis)


Guard = Annotated[LoginGuard, Depends(get_login_guard)]


@router.post(
    "/register",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[limit_register],
)
async def register(payload: UserCreate, db: DbSession):
    user_service.check_registration_allowed(payload.invite_code)
    return await user_service.register_user(db, payload.email, payload.password)


@router.post("/login", response_model=Token, dependencies=[limit_login])
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    request: Request,
    db: DbSession,
    limiter: Limiter,
    guard: Guard,
    response: Response,
):
    # OAuth2 form fields are named "username"/"password"; username holds the email.
    email, ip = user_service.normalize_email(form.username), client_ip(request)
    await enforce_login_email(limiter, email, response)
    # Before the (deliberately slow) password check: a blocked attempt costs
    # us almost nothing.
    await guard.check(email, ip)
    user = await user_service.authenticate(db, email, form.password)
    if user is None:
        await guard.failed(email, ip)
        # One message for both "no such email" and "wrong password".
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=INCORRECT_LOGIN,
            headers={"WWW-Authenticate": "Bearer"},
        )
    await guard.succeeded(email, ip)
    return Token(access_token=create_access_token(user.id))


@router.get("/me", response_model=UserRead, dependencies=[limit_default])
async def me(current_user: CurrentUser):
    return current_user
