# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See LICENSE for more details.
#
# Copyright (c) 2026 ScyllaDB

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from sdcm.utils import aws_okta


def _session_raising(error):
    session = MagicMock()
    session.client.return_value.get_caller_identity.side_effect = error
    return session


EXPIRED_TOKEN = ClientError(
    {"Error": {"Code": "ExpiredToken", "Message": "The security token included in the request is expired"}},
    "GetCallerIdentity",
)


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(EXPIRED_TOKEN, id="expired_token"),
        pytest.param(NoCredentialsError(), id="no_credentials"),
    ],
)
def test_a_login_that_is_not_valid_is_reported_rather_than_raised(error):
    """This is the question 'do I need to log in again' -- so every way of not being logged in has
    to answer it with False. An expired token is the commonest of them, and STS reports it as a
    ClientError rather than by failing to find credentials, so before it was handled the one case
    the Okta re-login exists for crashed the CLI instead of triggering it."""
    with patch.object(aws_okta.boto3, "Session", return_value=_session_raising(error)):
        assert aws_okta.can_get_to_aws_account() is False


def test_the_wrong_account_is_not_the_expected_one():
    """Valid credentials for some other account are not the login this asks about."""
    session = MagicMock()
    session.client.return_value.get_caller_identity.return_value = {"Account": "not-the-one", "Arn": "arn:whoever"}
    with patch.object(aws_okta.boto3, "Session", return_value=session):
        assert aws_okta.can_get_to_aws_account() is False


def test_the_expected_account_is_a_valid_login():
    session = MagicMock()
    session.client.return_value.get_caller_identity.return_value = {
        "Account": aws_okta.account_id,
        "Arn": "arn:aws:sts::x:assumed-role/y/z",
    }
    with patch.object(aws_okta.boto3, "Session", return_value=session):
        assert aws_okta.can_get_to_aws_account() is True
