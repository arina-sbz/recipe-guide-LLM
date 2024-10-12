#!/usr/bin/env python
# coding: utf-8

# In[2]:


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model = AutoModelForCausalLM.from_pretrained("VishalMysore/cookgptlama")
tokenizer = AutoTokenizer.from_pretrained("VishalMysore/cookgptlama")


# In[3]:


pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# system instruction
messages = [
    {
        "role": "system",
        "content": "You are an expert chef with vast knowledge of recipes from global cuisines. Your task is to provide users with detailed recipes, including ingredients, step-by-step cooking instructions, preparation time, and tailored dietary preferences or restrictions options.",
    }
]


# In[4]:


# handling a conversation with the model
def generate_response(pipe, messages, input):
    messages.append({"role": "user", "content": input})
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    outputs = pipe(prompt, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

    # get the assistant response from the output
    model_response = outputs[0]["generated_text"].split("<|assistant|>")[-1].strip()
    # print(model_response)

    # append the model's response to the messages
    messages.append({"role": "assistant", "content": model_response})

    return model_response, messages


# # In[4]:


# first_input = "Can you tell me how to make lasagna? I'm vegan."
# response, messages = generate_response(pipe, messages, first_input)
# print(response)


# # In[5]:


# second_input = (
#     "Can you suggest something to drink with the lasagna? I want alcohol-free."
# )
# response, messages = generate_response(pipe, messages, second_input)
# print(response)


# # In[6]:


# third_input = "What food recipe I asked for?"
# response, messages = generate_response(pipe, messages, third_input)
# print(response)


# # In[7]:


# print(messages)


# In[8]:


# # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
# messages = [
#     {
#         "role": "system",
#         "content": "You are an expert chef with vast knowledge of recipes from global cuisines. Your task is to provide users with detailed recipes, including ingredients, step-by-step cooking instructions, preparation time, and tailored dietary preferences or restrictions options.",
#     },
#     {"role": "user", "content": "Can you tell me how to make lasagna? I'm vegan."},
# ]
# prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# outputs = pipe(prompt, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
# # max_new_tokens=256,
# print(outputs[0]["generated_text"])


# In[9]:


# print(outputs)


# In[10]:


# model_output = outputs[0]["generated_text"]
# if "<|assistant|>" in model_output:
#     response = model_output.split("<|assistant|>")[-1]

# print(response.strip())


# In[11]:


# messages.append({"role": "system", "content": response})

# new_input = "Do you remember what you told me in your previous message? Can you suggest some thing to drink with the lasagna? I want alcohol free."

# messages.append({"role": "user", "content": new_input})
# print(f"-------{messages}------")

# prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# outputs = pipe(prompt, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

# print(outputs[0]["generated_text"].strip())


# In[24]:
