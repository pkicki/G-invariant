### Structure of the config files

##### Syntax
`--A=value_of_A`\
`--B=value_of_B`\
`...`

Notice: `value_of_X` must not contain whitespaces

#####Arguments

`--scenario-path` - path to the dataset\
`--working-path` - path to the directory where the logs and models will be stored
`--model` - name of the model
`--num-epochs` - number of epochs (int)
`--batch-size` - batch size (int)
`--log-interval` - breaks between logs (int)
`--out-name` - name of the directory where the logs and models will be stored
`--eta` - learning rate (float)
`--n` - size of the G-invariant latent vector (n_mid from the paper)
