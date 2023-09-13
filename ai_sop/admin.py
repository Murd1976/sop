from django.contrib import admin
from .models import AdvUser

class AdvUserAdmin(admin.ModelAdmin):
    search_fields = ['is_activated', 'send_messages', 'paid_account']

# Register your models here.
admin.site.register(AdvUser, AdvUserAdmin)