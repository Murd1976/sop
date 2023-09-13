from django.db import models
from django.utils import timezone
from django.contrib.auth.models import AbstractUser, Group, Permission
from django.utils.translation import gettext_lazy as _

version = 1.0

class AdvUser(AbstractUser):
    is_activated = models.BooleanField(default = True, db_index = True, verbose_name = 'Has been activated ?')
    send_messages = models.BooleanField(default = True, verbose_name = 'Send update messages ?')
    paid_account = models.BooleanField(default = False)
    '''
    groups = models.ManyToManyField(
        Group,
        verbose_name=_('groups'),
        blank=True,
        help_text=_(
            'The groups this user belongs to. A user will get all permissions '
            'granted to each of their groups.'
        ),
        related_name="advuser_set",
        related_query_name="advuser",
    )

    user_permissions = models.ManyToManyField(
        Permission,
        verbose_name=_('user permissions'),
        blank=True,
        help_text=_('Specific permissions for this user.'),
        related_name="advuser_set",
        related_query_name="advuser",
    )
    '''
    class Meta(AbstractUser.Meta):
        pass

class ChatList(models.Model):
    owner = models.ForeignKey(AdvUser, verbose_name='Chat owner', on_delete = models.DO_NOTHING)
    chat_name = models.CharField(max_length=100)
    
    created_at = models.DateTimeField(auto_now_add=True, db_index=True, verbose_name='Message time.')
    chat_type = models.CharField(max_length=30, default = 'none')
    company = models.CharField(max_length=100, default = 'none')
    subject = models.CharField(max_length=100, default = 'none')
    status = models.CharField(max_length=30, default = 'open')

class ChatHistory(models.Model):
    message = models.ForeignKey(ChatList, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True, verbose_name='Message time.')
    user_question = models.TextField()
    ai_answer = models.TextField()

    def __str__(self):
        return f"ChatHistory: {self.user_question} - {self.ai_answer}"



