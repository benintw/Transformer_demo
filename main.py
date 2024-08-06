import streamlit as st
import torch
import torch.nn as nn
from my_transformer import Transformer, config


def main():
    st.set_page_config(page_title="Transformer Architecture", layout="wide")

    st.title("Transformer Architecture: Input and Output Shapes")

    vocab_size = 50257
    transformer = Transformer(config, vocab_size)

    src = torch.randint(0, vocab_size, (config.batch_size, config.seq_len))
    tgt = torch.randint(0, vocab_size, (config.batch_size, config.seq_len // 2))

    st.subheader("Input Shapes (dummy data)")
    st.write(f"**Source (src):** `{src.shape}`")
    st.write(f"**Target (tgt):** `{tgt.shape}`")

    enc_output = transformer.encoder(
        transformer.embedding(src)
        * torch.sqrt(torch.tensor(config.d_model, dtype=torch.float32))
        + transformer.pos_encoding(src)
    )
    dec_output = transformer.decoder(
        enc_output,
        transformer.embedding(tgt)
        * torch.sqrt(torch.tensor(config.d_model, dtype=torch.float32))
        + transformer.pos_encoding(tgt),
    )

    st.subheader("Output Shapes")
    st.write(f"**Encoder Output:** `{enc_output.shape}`")
    st.write(f"**Decoder Output:** `{dec_output.shape}`")

    output = transformer(src, tgt)
    st.write(f"**Transformer Output:** `{output.shape}`")

    st.sidebar.header("Transformer Configuration")
    config_options = {
        "d_model": st.sidebar.slider("d_model", 128, 1024, config.d_model, step=64),
        "input_dim": st.sidebar.slider(
            "input_dim", 128, 1024, config.input_dim, step=64
        ),
        "num_heads": st.sidebar.slider("num_heads", 1, 16, config.num_heads),
        "num_layers": st.sidebar.slider("num_layers", 1, 12, config.num_layers),
        "batch_size": st.sidebar.slider("batch_size", 1, 64, config.batch_size),
        "seq_len": st.sidebar.slider("seq_len", 10, 500, config.seq_len, step=10),
        "dropout": st.sidebar.slider("dropout", 0.0, 0.5, config.dropout, step=0.05),
    }

    st.sidebar.write("### Updated Configuration")
    st.sidebar.json(config_options)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )

    num_epochs = st.sidebar.number_input(
        "Number of Epochs", min_value=1, max_value=100, value=2
    )

    if st.sidebar.button("Start Training"):
        transformer.train()

        loss_values = []
        progress_bar = st.progress(0)
        loss_chart = st.line_chart()

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = transformer(src, tgt[:, :-1])
            loss = criterion(
                output.contiguous().view(-1, vocab_size),
                tgt[:, 1:].contiguous().view(-1),
            )
            loss.backward()
            optimizer.step()

            loss_values.append(loss.item())
            progress_bar.progress((epoch + 1) / num_epochs)
            loss_chart.add_rows([loss_values])

            st.write(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

        st.success("Training complete!")
        st.write(f"Final Loss: {loss.item()}")


if __name__ == "__main__":
    main()
