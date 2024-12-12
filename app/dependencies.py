"""Application Dependencies"""

from fastapi.security import APIKeyHeader, OAuth2PasswordBearer


# api key authentication
api_key = APIKeyHeader(name="access_key")
api_key_multiple = APIKeyHeader(name="access_key", auto_error=False)

# authentication scheme
oauth2_scheme = OAuth2PasswordBearer("token")
oauth2_scheme_multiple = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

