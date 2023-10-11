from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse_lazy, reverse
from django.template import TemplateDoesNotExist
from django.template.loader import get_template
from django.contrib import messages
from django.contrib.messages.views import SuccessMessageMixin
from django.contrib.auth import logout
from django.contrib.auth.views import LoginView, LogoutView, PasswordChangeView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from django.views.generic.edit import UpdateView, CreateView, DeleteView
from django.views.generic.base import TemplateView
from django.utils.translation import gettext as _
from django.http import JsonResponse, HttpResponse
#from django.db.models import Q

import logging
import openai
import pandas as pd
from bs4 import BeautifulSoup
import re
import json

from django.views.generic import FormView

from wiki import editors
#from wiki import models as wiki_models
from wiki import forms as wiki_forms
from wiki.views.mixins import ArticleMixin
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from wiki.decorators import get_article
from wiki.conf import settings
from wiki.core import permissions
from mptt.models import MPTTModel

from ..forms import *
from ..models import *
from ..ai_sop import *
from ..ai_embedding import *


# Create your views here.
log = logging.getLogger(__name__)


company_dir = os.getenv('COMPANY_DIR') #'MagicVaporizers/'
#company_dir = 'Maxximize/'

my_url = "https://magicvaporizers.co.uk"

class BBLoginView(LoginView):
    template_name = 'sop/sop_login.html'
    
class BBLogoutView(LoginRequiredMixin, LogoutView):
    template_name = 'sop/sop_logout.html'

class RegisterDoneView(TemplateView):
    template_name = 'sop/sop_register_done.html'
    
class RegisterUserView(CreateView):
    model = AdvUser
    template_name = 'sop/sop_register_user.html'
    form_class = RegisterUserForm
    success_url = reverse_lazy('ai_sop:my_register_done')

class ChangeUserInfoView(SuccessMessageMixin, LoginRequiredMixin, UpdateView):
    model = AdvUser
    template_name = 'sop/sop_change_user_info.html'
    form_class = ChangeUserInfoForm
    success_url = reverse_lazy('ai_sop:my_profile')
    success_message = 'User data changed'
    
    def setup(self, request, *args, **kwargs):
        self.user_id = request.user.pk
        return super().setup(request, *args, **kwargs)
    
    def get_object(self, queryset=None):
        if not queryset:
            queryset = self.get_queryset()
        return get_object_or_404(queryset, pk=self.user_id)
    
class BBPasswordChangeView(SuccessMessageMixin, LoginRequiredMixin, PasswordChangeView):
    template_name = 'sop/sop_password_change.html'
    success_url = reverse_lazy('ai_sop:my_profile')
    success_message = 'User password changed'
    
class DeleteUserView(LoginRequiredMixin, DeleteView):
    model = AdvUser
    template_name = 'sop/sop_delete_user.html'
    success_url = reverse_lazy('ai_sop:index')
    
    def setup(self, request, *args, **kwargs):
        self.user_id = request.user.pk
        return super().setup(request, *args, **kwargs)
    
    def post(self, request, *args, **kwargs):
        logout(request)
        messages.add_message(request, messages.SUCCESS, 'User deleted')
        return super().post(request, *args, **kwargs)
    
    def get_object(self, queryset = None):
        if not queryset:
            queryset = self.get_queryset()
        return get_object_or_404(queryset, pk = self.user_id)
    
def index(request):   
    content = 'The homepage of AI Chat.'
    context = {"content": content}
    return render(request, "sop/index.html", context)

@login_required
def user_profile(request):
    return render(request, 'sop/sop_profile.html')

def user_activate(request, sign):
    try:
        
        username = signer.unsign(sign)
    except BadSignature:
        return render(request, 'sop/sop_bad_signature.html')
    
    user = get_object_or_404(AdvUser, username = username)
    if user.is_activated:
        template = 'sop/sop_user_is_activated.html'
    else:
        template = 'sop/sop_activation_done.html'
        user.is_active = True
        user.is_activated = True
        user.save()
    return render(request, template)# Create your views here.
    
    
# delete chat
def delete_chat(request, id):
    try:
        test = ChatList.objects.get(id=id)
        test.delete()
        #print('Deleted')
        return redirect('ai_sop:my_sop_chat')
    except ChatList.DoesNotExist:
        template = 'sop/sop_msg_page.html'
        context = {"my_msg":"Record not found"}
        return render(request, template, context)
        

    
def chat_page(request, id = 0):
    template = 'sop/sop_gpt_response.html'
        
    #company_dir = 'MagicVaporizers/'
    
    system_file = 'support_instruction.txt'
    
    try:
        gptLearning = WorkerОpenAIChat('chroma', system_file, company_dir)
    except openai.OpenAIError as e:
        template = 'sop/sop_msg_page.html'
        context = {"my_msg":f'OpenAI API Error: {e}'}
        return render(request, template, context)
    
    #chats = ChatList.objects.all()
    chats = ChatList.objects.filter(owner = request.user)
        
    subject_list = [(0, "DELIVERY INQUIRIES"), (1, "EDIT"), (2, "PRICE MATCHING"), (3, "PRODUCT INQUIRIES"), (4, "RETURNS"), (5, "Customer Service - Guide"), (6, "FAQ")]
    company_list = [(0, "Maxximize"), (1, "MagicVaporizers")]
    if request.method == "POST":
        
        userform = ChatForm(request.POST or None)
                        
        if userform.is_valid():
            
            query = userform.cleaned_data["f_ask_field"]
            chat_name = userform.cleaned_data["f_chat_name"]
            tmp = float(userform.cleaned_data['f_chat_temp'])
            print(f'\nTemperature: {tmp}\n')
                        
            full_query = {}
            full_query['company'] = "MagicVaporizers" #dict(company_list)[int(userform.cleaned_data["f_company"])]
            full_query['user_name'] = request.user.username
            print(f"\n UserName: {full_query['user_name']}\n")
            full_query['subject'] = dict(subject_list)[int(userform.cleaned_data["f_subject"])]
            full_query['issue'] = query
            
            gptLearning.question_history = []
            
            if(id > 0):
                curr_chat = ChatList.objects.get(id=id)
                print(f'\n ID: {id}\n ')
                res = ChatHistory.objects.filter(message=curr_chat)
                
                for qa in res:
                    if not (pd.isna(qa.user_question)):
                        gptLearning.question_history.append(('\n' + qa.user_question, qa.ai_answer if qa.ai_answer is not None else ''))
            
            
            try:
                response = gptLearning.answer_user_question(full_query, temp = tmp, verbose = 1)
            except openai.OpenAIError as e:
                template = 'sop/sop_msg_page.html'
                context = {"my_msg":f'OpenAI API Error: {e}'}
                return render(request, template, context)
                
            #print(f'\n User: {request.user} \n')
            
            if( id == 0):
                # сохраняем ответ в базу данных
                ChatList.objects.create(
                    owner = request.user,
                    chat_type = 'support',
                    company = full_query['company'],
                    chat_name = make_chat_name(query),
                    subject = full_query['subject']
                )
                curr_chat = ChatList.objects.last()
                # создаем новый чат для нового предложения и сохраняем в базу данных
                ChatHistory.objects.create(
                    message = curr_chat,
                    user_question = query,
                    ai_answer = response
                )
            else:
                
                # создаем новый чат для нового предложения и сохраняем в базу данных
                ChatHistory.objects.create(
                    message = curr_chat,
                    user_question = query,
                    ai_answer = response
                )
            
            messages = ''
            res = ChatHistory.objects.filter(message=curr_chat)
                    
            for qa in res:
                if not (pd.isna(qa.user_question)):
                    messages += '\n\n User: ' + qa.user_question + ' ' + '\n Assistant: \n' + (qa.ai_answer if qa.ai_answer is not None else '')
            
            
            st = ""
            for i in range(len(gptLearning.debug_log)):
                st += str(gptLearning.debug_log[i])
                        
           # print(request.headers)
            #print(response)
            '''
            messages = '\n\n User: ' + query + ' ' + '\n Assistant: \n' + response
            parts = ChatForm(initial= {"f_chat_field":messages, 'f_debug_field':st})
            
            context = {"form": parts, "chat_list": ''}
            return render(request, template, context)
            '''
            # Формируем URL-адрес с параметром запроса
            redirect_url = reverse('ai_sop:my_sop_chat_id', kwargs={'id': curr_chat.id})
            #redirect_url = reverse('ai_sop:my_sop_chat_id', kwargs={'id': 0})
            redirect_url += f'?debug_field={st}'
            return redirect(redirect_url)
            
            #return redirect('chat:my_chat_gpt_id', id=curr_chat.id)
        else:
            template = 'sop/sop_msg_page.html'
            context = {"my_msg":"Something was wrong..."}
            return render(request, template, context)
            
    messages = ''
    if (id > 0): 
        curr_chat = ChatList.objects.get(id=id)
    else:
    
        curr_chat = None
            
    if (curr_chat == None):
        parts = ChatForm()
    else:
        
        res = ChatHistory.objects.filter(message=curr_chat)
                    
        for qa in res:
            if not (pd.isna(qa.user_question)):
                messages += '\n\n User: ' + qa.user_question + ' ' + '\n Assistant: \n' + (qa.ai_answer if qa.ai_answer is not None else '')
        additional_data = request.GET.get('debug_field', None)
        
        parts = ChatForm(initial= {"f_company":get_key_by_val(curr_chat.company, company_list), "f_subject":curr_chat.subject, "f_chat_name":curr_chat.chat_name, "f_chat_field":messages, 'f_debug_field':additional_data})
        
    
    context = {"form": parts, "chat_list": chats}
    return render(request, template, context)
    
def get_key_by_val(val, source):
    
    for k, v in source:
        
        if v == val:
            return k
            
def make_chat_name(query) -> str:
    word_list = query.split()
    name = ''
    if (len(word_list) > 5):
        for i in range(5):
            name += ' ' + word_list[i]
    else:
        for i in range(len(word_list)):
            name += ' ' + word_list[i]
            
    return name
    
def chat_instruction(request):
    template = 'chat/chat_gpt_instruction.html'
    # путь к учебным материалам
    #data_directory = 'chat/db/Cris/'
    #data_directory = 'chat/db/MagicVap_sys/'
    #data_directory = 'chat/db/chat/sys/Maxximize/'
    data_directory = 'chat/db/chat/sys/' + company_dir
    system_doc = 'support_instruction.txt'
    
    if request.method == "POST":
        #text_buf = "Amswer: "
        userform = EditInstructionForm(request.POST or None)
        #text_buf += dict(strategies_value)[str(userform.data.get("f_strategies"))]
                
        if userform.is_valid():
            txt = userform.cleaned_data['f_chat_instruction']
            print('\n', txt)
            f = open(data_directory + system_doc, 'w')
            f.write(txt)
            f.close()
    try:
        with open(data_directory + system_doc, "r") as f:
            text = f.read()
        f.close()
    except:
        text = "Instruction not found."
    
    parts = EditInstructionForm(initial = {'f_chat_instruction': text})
    context = {"form": parts}
    return render(request, template, context)
    
def prop_instruction(request):
    template = 'chat/chat_gpt_instruction.html'
    # путь к учебным материалам
    #data_directory = 'chat/db/CrisProposal/'
    #data_directory = 'chat/db/MagicVap_sys/'
    data_directory = 'chat/db/proposal/sys/Maxximize/'

    system_doc = 'proposal_instruction.txt'
    
    if request.method == "POST":
        #text_buf = "Amswer: "
        userform = EditInstructionForm(request.POST or None)
        #text_buf += dict(strategies_value)[str(userform.data.get("f_strategies"))]
                
        if userform.is_valid():
            txt = userform.cleaned_data['f_chat_instruction']
            print('\n', txt)
            f = open(data_directory + system_doc, 'w')
            f.write(txt)
            f.close()
            
    try:
        with open(data_directory + system_doc, "r") as f:
            text = f.read()
        f.close()
    except:
        text = "Instruction not found."
    
    parts = EditInstructionForm(initial = {'f_chat_instruction': text})
    context = {"form": parts}
    return render(request, template, context)
    

def handle_uploaded_file(f):
    with open('some/file/name.txt', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    
def contains_html_tags(text):

    pattern = re.compile('<[^>]+>')
    
    return bool(pattern.search(text))

def div_title_content(text):
    html_content = BeautifulSoup(text, 'html.parser')
    # Находим тег <h1>
    h1_tag = html_content.find('h1')
    if h1_tag:
        # Получаем содержимое тега <h1>
        title = h1_tag.string
        h1_tags = html_content.find_all('h1')
        
        # Удаляем тег <h1>
        # Удаляем каждый найденный тег
        for h1 in h1_tags:
            h1.decompose()
            
        content = str(html_content)
    else:
        title = list(text)[0]
        
        content = text #.split('\n', 1)[1]
        
    print(f' Make title: {title}')
    

    return title, content
    
def get_ai_content(f_name, mode = 'usal', instructions = ''):
    #res_f_name = 'magicvaporizers.txt'
        
    #res_f_name = 'magicvaporizers.txt'
    #my_url = "https://magicvaporizers.co.uk"
        
    gptLearning = WorkerОpenAI_SOP('chroma', '', company_dir)
    
    #text = scrape_site(my_url, res_f_name)
    #text = gptLearning.web_text_filter_2('magicvaporizers_short.txt')
    
    if mode == 'usal':
        prompt_template = """The following provides context to the raw information associated with the company's operations.
    
    The information contained in this original guide in [Context] needs to be structured and converted into an HTML document or HTML article for Wikipedia.
    Headings at different levels should be highlighted using HTML tags.
    The first line should be a title reflecting the essence of the document.
            
    {user_rules}
      
    Context: {context}
    
    Analysis result:"""
    elif mode == 'sop':
        prompt_template = """Your job is to create an SOP (standard operating procedures) document.. Below, between >>> and <<<, is the context of the raw information associated with the business of Magic Vaporizers.
        
        This can be the whole document or part of it.
              
        Using  this raw guide, create a template of a Customer Support SOP document for an ecommerce company. 
        Responce should be formated with HTML tags.
                              
        Format and style: the SOP must be written in the form of structured instructions, with step by step actions to perform,
        organized in flowchart structures, and also with examples of customer questions/tickets and the agent answers.
        A statement in the form of action steps are a numbered list.
        Example:
            Name of the section to which the current instruction belongs
            Instruction name
            1. First step
                 - Substep
                 - Substep
            2. Second sterp
            
        Name of headers, sections and instruction - are not numbered!
        Instead of the title: Section 1: Invoice Request, it should be - Invoice Request
        Instead of the title: Instruction name: Find Order, it should be Find Order

        Instead of headings: I. Delivery Inquiries,  A.Overview,  B. Process,   
        should be headings: Delivery Inquiries, Overview, Process
                       
        {user_rules}
                                   
        Context: >>>{context}<<<
        
        Analysis result:"""
        
    
    elif mode == 'faq':
        prompt_template = ''
    elif mmode == 'html':
        prompt_template = """The text must be structured and converted into an HTML document.
    Headings of different levels, lists and links must be highlighted using HTML tags.
    Links should be highlighted in color, as on a standard WEB page.
    Each element of a numbered list must begin on a new line.
    All first level lists must be numbered digitally.
    
    Analysis result:"""
    else:
        prompt_template = ''
        
    #'FAQ _ Returns DryHerbs.docx'
    text = gptLearning.doc_analyzer_sop(f_name, prompt_template, instructions)
    
    return div_title_content(text)
    
class Create_Article_Page(FormView, ArticleMixin):
    form_class = AiCreateForm
    template_name = "sop/create_article.html"

    @method_decorator(get_article(can_write=True, can_create=True))
    def dispatch(self, request, article, *args, **kwargs):
        return super().dispatch(request, article, *args, **kwargs)
    
    def get_form(self, form_class=None):
        """
        Returns an instance of the form to be used in this view.
        """
        if form_class is None:
            form_class = self.get_form_class()
        kwargs = self.get_form_kwargs()
        initial = kwargs.get("initial", {})
        initial["slug"] = self.request.GET.get("slug", None)
        kwargs["initial"] = initial
        form = form_class(self.request, self.urlpath, **kwargs)
        form.fields["slug"].widget = wiki_forms.TextInputPrepend(
            prepend="/" + self.urlpath.path,
            attrs={
                # Make patterns force lowercase if we are case insensitive to bless the user with a
                # bit of strictness, anyways
                "pattern": "[a-z0-9_-]+"
                if not settings.URL_CASE_SENSITIVE
                else "[a-zA-Z0-9_-]+",
                "title": "Lowercase letters, numbers, hyphens and underscores"
                if not settings.URL_CASE_SENSITIVE
                else "Letters, numbers, hyphens and underscores",
            },
        )
        #self.creation(form)
        return form
    
    def form_valid(self, form):
        file_list = [(0, "SOP Dry Herbs Vaporizer - V20211031.docx"), (1, "Template Messages - MagicVaporizers.docx"), (2, "FAQ _ Returns DryHerbs.docx")]
        
        f_name = dict(file_list)[int(form.cleaned_data["file_name"])]
        #handle_uploaded_file(request.FILES['file'])
        title = form.cleaned_data["title"]
        instructions = form.cleaned_data["instructions"]
        slug = form.cleaned_data["slug"]
        content = ''
        summary = ''
        print(f'\n Path: {self.urlpath} \n')
        if str(self.urlpath).strip() == 'sop_/' :
            title_new, content = get_ai_content(mode = 'sop', f_name = f_name, instructions = instructions)
            
            summary = '' #get_ai_summary(content)
        else:
            title_new, content = get_ai_content(mode = 'usal', f_name = f_name, instructions = instructions)
            summary = '' #get_ai_summary(content)
        print(content)    
        try:
            self.newpath =wiki_models.URLPath._create_urlpath_from_request(
                self.request,
                self.article,
                self.urlpath,
                slug,
                title,
                content,
                summary,
            )
            messages.success(
                self.request,
                _("New article '%s' created.")
                % self.newpath.article.current_revision.title,
            )
        # TODO: Handle individual exceptions better and give good feedback.
        except Exception as e:
            log.exception("Exception creating article.")
            if self.request.user.is_superuser:
                messages.error(
                    self.request,
                    _("There was an error creating this article: %s") % str(e),
                )
            else:
                messages.error(
                    self.request, _("There was an error creating this article.")
                )
            return redirect("wiki:get", "")

        return self.get_success_url()
    '''
    def creation(self, form):
        #print('\n Creating... \n')
        try:
            
            self.newpath = wiki_models.URLPath._create_urlpath_from_request(
                self.request,
                self.article,
                self.urlpath,
                form.cleaned_data["slug"],
                'My new article', #form.cleaned_data["title"],
                'Content of new article', #form.cleaned_data["content"],
                '', #form.cleaned_data["summary"],
            )
            messages.success(
                self.request,
                _("New article '%s' created.")
                % self.newpath.article.current_revision.title,
            )
            print(f'\n Article: {self.article}  Urlpath: {self.urlpath}')
        # TODO: Handle individual exceptions better and give good feedback.
        except Exception as e:
            log.exception("Exception creating article.")
            if self.request.user.is_superuser:
                messages.error(
                    self.request,
                    _("There was an error creating this article: %s") % str(e),
                )
            else:
                messages.error(
                    self.request, _("There was an error creating this article.")
                )
            return redirect("wiki:get", "")

        return self.get_success_url()
     '''   
    def get_success_url(self):
        return redirect("wiki:get", self.newpath.path)

    def get_context_data(self, **kwargs):
        c = ArticleMixin.get_context_data(self, **kwargs)
        c["form"] = self.get_form()
        c["parent_urlpath"] = self.urlpath
        c["parent_article"] = self.article
        c["create_form"] = c.pop("form", None)
        c["editor"] = editors.getEditor()
        return c
        
#============================================================================================================
#----------------------------------------------------------------------------
class SopView(ArticleMixin, TemplateView):

    template_name = "wiki/view.html"

    @method_decorator(get_article(can_read=True))
    def dispatch(self, request, article, *args, **kwargs):
        #print(my_tree_to_json())
        try:
            
            self.root = wiki_models.URLPath.get_by_path("")
            self.three = wiki_models.URLPath.get_ordered_children(self.root)
            for three_elem in self.three:
                #print(f'Three: {three_elem}')
                if (str(three_elem).strip() in ['sop/', 'sop_/']):
                    #messages.success( self.request,  _("The SOP section already exist."))
                    self.current_path = three_elem
                    
                    return self.get_success_url()
             
            try:
                self.newpath = wiki_models.URLPath._create_urlpath_from_request(
                    self.request,
                    article,
                    self.root,
                    'sop_',
                    "SOP",
                    'All documents containing Standard Operating Procedures (SOP)',
                    '',
                )
                        
                messages.success(
                            self.request,
                            _("The SOP section was created.")
                    
                        )
                self.current_path = self.newpath
                return self.get_success_url()
                    
            # TODO: Handle individual exceptions better and give good feedback.
            except Exception as e:
                log.exception("The SOP not created!")
                if self.request.user.is_superuser:
                    messages.error(
                        self.request,
                        _("There was an error creating SOP section: %s") % str(e),
                    )
                else:
                    messages.error(
                        self.request, _("There was an error creating SOP section.")
                    )
                return redirect("wiki:get", "")
        except:
            messages.error(
                        self.request, _("Can't read three of wiki.")
                    )
            return redirect("wiki:get", "")
    
        return super().dispatch(request, article, *args, **kwargs)
        
    def get_success_url(self):
        return redirect("wiki:get", self.current_path.path)

    def get_context_data(self, **kwargs):
        kwargs["selected_tab"] = "view"
        return ArticleMixin.get_context_data(self, **kwargs)
        
        
    def auto_create_article(self, request, article, urlpath, slug, title, content, summary):
    
        try:
            
            self.current_path = wiki_models.URLPath._create_urlpath_from_request(
                request,
                article,
                urlpath,
                slug,
                title,
                
                content,
                summary,
            )
            
            print(f'\n Article: {self.article}  Urlpath: {self.urlpath}')
        # TODO: Handle individual exceptions better and give good feedback.
        except Exception as e:
            log.exception("Exception creating article.")
            if self.request.user.is_superuser:
                messages.error(
                    self.request,
                    _("There was an error creating this article: %s") % str(e),
                )
            else:
                messages.error(
                    self.request, _("There was an error creating this article.")
                )
            return redirect("wiki:get", "")

        return self.get_success_url()

    def create_list_articles(self, request, urlpath, list_article):
        
        summary = ''
        for elem in list_article:
            new_url = self.auto_create_article(request, self.article, urlpath, elem.title, elem.title, elem.content, summary)
            
class AssistantView(ArticleMixin, TemplateView):

    @method_decorator(csrf_exempt)  # Отключите CSRF проверку (в реальной жизни лучше настроить CSRF)
    @method_decorator(get_article(can_write=True, can_create=True))    
    def dispatch(self, request, article, *args, **kwargs):
        #print("Got request to AI.")
        return super().dispatch(request, article, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        # Получение данных с сайта
        instruction = request.POST.get('instruction')
        article_id = request.POST.get('article_id')
        orign_content = self.article.current_revision.content
        orign_title = self.article.current_revision.title
        print(f" FeedBack: {instruction} Article title: {orign_title}\n\n Article content: {orign_content}\n")
        
        # Обработка данных
        #processed_text = f" FeedBack: {instruction} Article title: {title}\n\n Article content: {content}\n" #'Returned text from server!'
        system_file = 'edit_instruction.txt'
        
        try:
            gptLearning = WorkerОpenAI_SOP('chroma', system_file, company_dir)
        except openai.OpenAIError as e:
            messages.error( self.request, _("OpenAI API Error: %s") % str(e), )
                        
        query = f" FeedBack: {instruction} Article title: {orign_title}\n\n Article content: {orign_content}\n"
            
        try:
            response = gptLearning.get_gpt_proposal(query, verbose = 1)
        except openai.OpenAIError as e:
            messages.error( self.request, _("OpenAI API Error: %s") % str(e), )
                   
        #print(type(response))
        print(response)
        res_data = {}
        
        try:
            json_str = response.replace('\n', ' ')
            #print(f'\n Replaced: {json_str} \n')
            res_data = json.loads(json_str) 
                        
            print('json.loads Ok!')
        except json.JSONDecodeError as e:
            #messages.error( self.request, _("JSON Error: %s") % str(e), )
            print(f"JSON Error: {e}")
            res_data = {
                'title': '',
                'content': '',
                'summary': ''
            }
            res_data["title"] = orign_title
            res_data["content"] = response
            res_data["summary"] = ''
        # Отправка результатов на сайт
       # print(res_data["content"], '\n')
       
        print(type(res_data))
        print('=================================================================================')        
        print(res_data["title"], '\n')
        print(res_data["content"], '\n')
        print(res_data["summary"])
        #import pdb; pdb.set_trace()
        return JsonResponse(res_data, safe=False)
        #return JsonResponse({'text': processed_text })
        #return HttpResponse(content_text)