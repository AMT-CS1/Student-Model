import torch
import torch.nn as nn
import warnings


warnings.filterwarnings('ignore')


class HELP_DKT_Model(nn.Module):
    """HELP-DKT model"""

    # Initialize model
    def __init__(self, rnn_type, args, num_skills, timeSteps, dropout=0.6, tie_weights=False):
        super(HELP_DKT_Model, self).__init__()
        # initialize dropout
        self.drop = nn.Dropout(dropout)
        
        self.rnn = getattr(nn, rnn_type)(args.input_size, args.hidden_size, args.hidden_layer_num, batch_first=True, dropout=dropout)
        
        # initialize model decoder sesuai dengan types task 
        
        # task A liner nn dengan hidden size*timestep 1 dimensi output
        if args.taskModel == 'taskA':
            self.decoder = nn.Linear(args.hidden_size*timeSteps, 1)
        
        # task b liner nn dengan hidden size difault dan 1 dimensi output
        elif args.taskModel == 'taskB':
            self.decoder = nn.Linear(args.hidden_size, 1)
        
        # task c sequence dengan linear dan sigmoid karena predict
        elif args.taskModel == 'taskC':
            
            self.decoder1 = nn.Sequential(

                nn.Linear(args.hidden_size, args.Qmatrix_size),
                nn.Sigmoid()
            )
            self.decoder2 = nn.Sequential(
                nn.Sigmoid()
            )

        # default initiate
        self.rnn_type = rnn_type
        self.nhid = args.hidden_size
        self.nlayers = args.hidden_layer_num
        self.taskModel = args.taskModel
        self.timeSteps = timeSteps
        self.multiLinearLayers = args.multiLinearLayers
        self.masked = args.masked
        self.QmatrixSize = args.Qmatrix_size
        self.subQmatrix = args.subQmatrix
        self.linearWithQmatrix = args.linearWithQmatrix

        self.init_weights()

    # awal weight initialize
    def init_weights(self):
        if self.multiLinearLayers == 'False':
            # 1 decoder linear
            initrange = 0.05
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)
        else:
            # double decoder
            if self.taskModel == 'taskA' or self.taskModel == 'taskB':
                
                for name, param in self.decoder.named_parameters():
                    if 'weight' in name:
                        initrange = 0.05
                        nn.init.uniform_(param, -initrange, initrange)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
            else:
                for name, param in self.decoder1.named_parameters():
                    if 'weight' in name:
                        initrange = 0.05
                        nn.init.uniform_(param, -initrange, initrange)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                for name, param in self.decoder2.named_parameters():
                    if 'weight' in name:
                        initrange = 0.05
                        nn.init.uniform_(param, -initrange, initrange)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    # for train
    def forward(self, input, hidden, Qmatrix, problemQmatrixMask,
                problemQmatrixSub, problemQmatrixAbilityMask, problemQmatrixProd, test=False):
        '''
        @description: run model
        @demands:
        @params:
        @return:
        '''
        # hasil hidden state ht dan 
        output, hidden = self.rnn(input, hidden)

        if self.multiLinearLayers == 'False':
            # Kalau False
            # Model menjadi DKT biasa.
            # Output LSTM langsung diprediksi.

            if self.taskModel == 'taskA':
                # MENYELESAIKAN DALAM T TIMESTAMP
                decoded = self.decoder(
                    output.reshape(-1, self.timeSteps*self.nhid))
            elif self.taskModel == 'taskB':
                # APAKAH NEXT SOAL BISA SELESAI
                decoded = self.decoder(output)
            else:
                raise ValueError('model forward ERROR!')
        else:
            # VERSI FULL HELP-DKT

            # decoded1 itu hasil dari lstm menjadi St dengan linear
            # decoder1 dari hidden size(output) menjadi Q-matrix size
            # yang berarti matriks pengetahuan
            decoded1 = self.decoder1(output)
            # inisiasi saja
            ability = decoded1

            # ini proses masking Qmatrix
            if self.masked == 'True':
                decoded1 = torch.mul(decoded1, problemQmatrixAbilityMask)
            
            # ini proses pengurangan dari hasil masking Qmatrix
            if self.subQmatrix == 'True':
                decoded1 = torch.sub(decoded1, problemQmatrixSub)

            # Torch prod itu perkalian total dalam hasil
            # atau perkalian antar kemampuan yang sudah
            # di sigmoidkan
            # baca code dari dalam prod dulu
            decoded = torch.prod(
                # decoder 2 itu sigmoid saja
                # menghasilkan probabilitas setiap kemampuan
                self.decoder2(
                    # multiplication dengan Aplha
                    torch.mul(decoded1, problemQmatrixProd)), dim=2, keepdim=True)

        if test == False:
            # ini kalau Training
            # decoded akan digunakan untuk loss
            return decoded, hidden
        else:
            # ini kalau test true

            # buat matriks all 1
            tmp = torch.ones_like(problemQmatrixAbilityMask)
            
            # Mengecek konsep yang tidak digunakan
            # Jika suatu timestep
            # tidak memiliki konsep
            # supaya nanti tidak divisualisasikan.
            for i in range(problemQmatrixAbilityMask.size()[0]):
                for j in range(problemQmatrixAbilityMask.size()[1]):
                    if 1 not in problemQmatrixAbilityMask[i, j, :]:
                        tmp[i, j, :] = 0

            if self.multiLinearLayers == 'True':
                # apa yang direturn
                # decoded
                # hidden layer
                # student ability yang 
                # dikalikan keperluan/ ability yang ada
                # dan dikembalikan ke dalam list
                return decoded, hidden, ability.mul(tmp).tolist()
            else:
                # ini kalau task A atau B
                return decoded, hidden, tmp.tolist()

    # initialize hidden statenya menjadi 0
    def init_hidden(self, bsz):
    # param bsz adalah batch size
        # menandai salah satu param model seprti device dan dtype
        weight = next(self.parameters())

        if self.rnn_type == 'LSTM':
            # return 2 tensor untuk hidden state dan cell state menjadi 0 
            # yang sesuai dengan jumlah hidden layer, batch
            # dan hidden size
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            # if rnn
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
