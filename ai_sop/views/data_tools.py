from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse_lazy, reverse
from django.template import TemplateDoesNotExist
from django.template.loader import get_template
from django.contrib import messages
from django.contrib.auth.decorators import login_required

from ..forms import *
from ..ai_sop import *
from ..ai_embedding import *


company_dir = os.getenv('COMPANY_DIR')

@login_required
def tools_page(request):   
    print('URL is working!')
    content = 'The homepage of AI Chat.'
    context = {"content": content}
    return render(request, "sop/sop_data_tools.html", context)

@login_required
def create_chat_embedding_page(request):

    f_name = "train_data_ask.jsonl"
    # папка базы данных
    #persist_directory = 'ai_sop/db/CrisEmbeddingsChat/'
    # путь к учебным материалам
    #data_directory = 'chat/db/Cris/'
    #"chat/train_data_ask.jsonl"
    system_doc = 'support_instruction.txt'
    
    if request.method == "POST":
        userform = DbLoadForm(request.POST or None)
        if userform.is_valid():
            f_name = userform.cleaned_data["db_name"]
            gptLearning = WorkerОpenAIChat('', company_dir)
    
            # # Подготовка эмбедингов
            gptLearning.create_embedding(gptLearning.data_directory, # путь к учебным материалам
                                        gptLearning.persist_directory) # путь для сохранения базы данных
        
    parts = DbLoadForm(initial= {"db_name":f_name})
    template = 'sop/sop_create_embedding.html'
    
    context = {"form": parts, "user_name":request.user}
    return render(request, template, context)
    
@login_required
def create_prop_embedding_page(request):

    f_name = "train_data_ask.jsonl"
    # папка базы данных
    #persist_directory = 'ai_sop/db/CrisEmbeddingsChat/'
    # путь к учебным материалам
    #data_directory = 'chat/db/Cris/'
    #"chat/train_data_ask.jsonl"
    system_doc = 'support_instruction.txt'
    
    if request.method == "POST":
        userform = DbLoadForm(request.POST or None)
        if userform.is_valid():
            f_name = userform.cleaned_data["db_name"]
            gptLearning = WorkerОpenAI_SOP('', company_dir)
    
            # # Подготовка эмбедингов
            gptLearning.create_embedding(gptLearning.data_directory, # путь к учебным материалам
                                        gptLearning.persist_directory) # путь для сохранения базы данных
        
    parts = DbLoadForm(initial= {"db_name":f_name})
    template = 'sop/sop_create_embedding.html'
    
    context = {"form": parts, "user_name":request.user}
    return render(request, template, context)
    
@login_required
def load_chat_data_page(request):

    f_name = "train_data_ask.jsonl"
    if request.method == "POST":
        userform = DbLoadForm(request.POST or None)
        if userform.is_valid():
            f_name = userform.cleaned_data["db_name"]
            load_from_xls(f_name, request.user)
    parts = DbLoadForm(initial= {"db_name":f_name})
    template = 'chat/chat_load_chat_base.html'
    chain_list = MessageChain.objects.filter(owner=request.user)

    context = {"form": parts, "tests_log": chain_list, "user_name":request.user}
    return render(request, template, context)

@login_required
def load_base_page(request):
    f_name = "chat/upwork_list.xlsx"
    if request.method == "POST":
        userform = DbLoadForm(request.POST or None)
        if userform.is_valid():
            f_name = userform.cleaned_data["db_name"]
            load_from_xls(f_name, request.user)
    parts = DbLoadForm(initial= {"db_name":f_name})
    template = 'chat/chat_load_base.html'
    chain_list = MessageChain.objects.filter(owner=request.user)
#    tests_list = AllBackTests.objects.all()
#    tests_list.delete()
#    tests_list = AllBackTests.objects.all()
    context = {"form": parts, "tests_log": chain_list, "user_name":request.user}
    return render(request, template, context)

def load_web_page(request):
    res_f_name = 'magicvaporizers.txt'
    my_url = "https://magicvaporizers.co.uk"
    
    #res_f_name = 'magicvaporizers.txt'
    #my_url = "https://magicvaporizers.co.uk"
    
    company_dir = 'MagicVaporizers/'
    #company_dir = 'Maxximize/'
    
    gptLearning = WorkerОpenAI_SOP('', company_dir)
    
    #text = scrape_site(my_url, res_f_name)
    #text = gptLearning.web_text_filter_2('magicvaporizers_short.txt')
    text = gptLearning.doc_analyzer_fields()
    #text = gptLearning.doc_analyzer_sop('FAQ _ Returns DryHerbs.docx')
    parts = EditInstructionForm(initial = {'f_chat_instruction': text})
    template = 'chat/chat_load_web.html'
    #context = {"form": parts, "tests_log": chain_list, "user_name":request.user}
    context = {"form": parts, "user_name":request.user}
    return render(request, template, context)