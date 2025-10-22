FROM whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6-mcore0.12.0-te2.3
# Set working directory
WORKDIR /opt

# Install verl
ENV VERL_COMMIT=2ed63bbf39c22724e4940d97e4b09e4f3e5f6d68
RUN git clone https://github.com/volcengine/verl.git && \
    cd verl && \
    git checkout ${VERL_COMMIT} && \
    pip3 install -e .

RUN pip install fire
RUN pip3 install -U pynvml


WORKDIR /workspace

# Fix CV2
RUN pip install opencv-fixer==0.2.5 && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"

# Run additional dependencies
RUN pip install math-verify[antlr4_9_3] ray[default] pylatexenc wandb

RUN pip install numba
RUN pip install peft
RUN pip install bitsandbytes

CMD ["/usr/bin/bash"]
WORKDIR /workspace