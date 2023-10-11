import pandas as pd
import numpy as np
from django.template.loader import render_to_string
from django.core.signing import Signer
import os

from sop.settings import ALLOWED_HOSTS

signer = Signer()

def send_activation_notification(user):
    if ALLOWED_HOSTS:
        host = 'http://' + ALLOWED_HOSTS[0]
    else:
        host = 'http://localhost:8000'
    context = {'user':user, 'host':host, 'sign':signer.sign(user.username)}
    subject = render_to_string('sop/email/sop_activation_letter_subject.txt', context)
    body_text = render_to_string('sop/email/sop_activation_letter_body.txt', context)
    user.email_user(subject.replace('\n', ''), body_text)

def get_upload_path(instance, filename):
    
    return 'ai_sop/db/data/{}/{}'.format(os.getenv('COMPANY_DIR'), filename)