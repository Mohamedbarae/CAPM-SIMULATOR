
from math import sqrt
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

actions=['MSFT' ,'WMT','BAC','CAT'] 
marché = ['^SP500TR']

noms = ['Microsoft' ,'Walmart','Bank of America','Caterpillar']  

actif_sans_risque = ['^IRX']     # les bons de trésors américains à 3 mois 
P=actions + marché

#=========================================== PARTIE 1 : ANALYSE DES ACTIONS ET DU MARCHÉ =================================================================================


# Téléchargement des cours boursiers ajustés (aux dividendes et aux splits) des actions : Microsoft , Walmart , Bank of America , Caterpillar et de l'indice S&P 500 Total Return .
def télécharger_et_afficher_les_données (valeurs):

    cours=yf.download(valeurs,period='5y',interval='1mo',auto_adjust='True')

    df=cours['Close'].round(2)[valeurs]
    print('=========== COURS AJUSTÉS AUX DIVIDENDES ET AUX SPLITS :========')
    print(df.to_string())
    return df


# Calcul des rendements mensuels :
def calculer_les_rendements_mensuels(df):
    rendements=df.pct_change().dropna()
    print('\n=============RENDEMENTS MENSUELS EN % :==========')
        
    print(rendements.round(2)*100)

    return rendements


# Calcul des rentabilités moyennes et les volatilités : 
def calculer_les_moyennes_et_volatilités(rendements):
    moyennes=rendements.mean()
    volatilités =rendements.std()
    

    return moyennes,volatilités

def annualiser_stats(moyennes,volatilités):
    moyennes_annualisées = (1+moyennes)**12-1
    print('=========== RENTABILITÉS ANNUALISÉES EN % :========')
    print(moyennes_annualisées.round(2)*100)


    volatilités_annualisées=volatilités*sqrt(12)
    print('============ VOLATILITÉS ANNUALISÉES EN % :========')
    print(volatilités_annualisées.round(2)*100)
    return moyennes_annualisées,volatilités_annualisées

# RECUPERATION DU TAUX SANS RISQUE : 
def importer_et_calculer_le_taux_sans_risque(sr):
    rf =yf.download(sr,period='5y',interval='1mo')
    irx=(rf['Close'])

    rf=(irx.mean().iloc[0])/100
    
    

    print('Le taux sans risque est de:' ,round(rf,2)*100, '%')
    return rf




# CALCUL DES BÊTAS : 
def calculer_les_betas(rendements):
    market = rendements['^SP500TR']
    BETA={}
    for action in actions:
        beta = rendements[action].cov(market) / market.var()
        print('Le bêta de l action ',action,'est : ',round(beta,2))
        BETA.update({action:beta})

    
    return BETA








# RENTABILITÉS ESPÉRÉES PAR LE MEDAF :


def calculer_les_rentabilités_espérées_par_le_MEDAF(rf, BETA, moyennes):

    Medafs=[]
    
    Rm=moyennes['^SP500TR']
    print('============ RENTABILITÉS ESPERÉES SELON LE MEDAF : :=========')
    for i in  actions :
        R = (rf+BETA[i]*(Rm-rf))
        Ra=round(R,2)
        print('La rentabilité espérée de l actif ',i,'est : ',Ra*100 ,'%')
        Medafs.append(R)
    return Medafs , Rm


 
# Calcul des ratio de sharpe pour chaque action : 

def calculer_les_ratios_de_sharpe(moyennes, volatilités, rf):
    Ratios_sharpe=[]
    print('\n============ RATIOS DE SHARPE :=========')
    for i in  actions :
        S = (moyennes[i] - rf)/ (volatilités[i])
        Sa=round(S,2)
        print('Le ratio de sharpe de ',i,'est : ',Sa )
        Ratios_sharpe.append(S)
    return Ratios_sharpe


def calculer_les_Alphas_de_Jensen(Medafs: list, moyennes: dict):
    Alphas=[]
    print('\n============ ALPHAS DE JENSEN :=========')
    for i in  actions :
        A = moyennes[i] - Medafs[actions.index(i)]
        Aa=round(A,2)
        print('L alpha de Jensen de ',i,'est : ',Aa )
        Alphas.append(A)
    return Alphas






# Simulations monte carlo pour avoir la meilleure pondération maximisant le ratio de sharpe :

def simulation_portefeuilles(rendements, moyennes_annualisées, rf,n):

    matrice_cov = rendements[actions].cov() * 12   # annualisation de la matrice de covariance
    resultats = []

    for i in range(n):

        poids = np.random.random(len(actions))
        poids /= poids.sum()

        rendement = np.sum(poids * moyennes_annualisées[actions])
        volatilite = np.sqrt(poids.T @ matrice_cov @ poids)

        sharpe = (rendement - rf) / volatilite

        resultats.append([rendement, volatilite, sharpe, *poids])
    colonnes = (["Rentabilité", "Volatilité", "Sharpe"]+ actions)
    return pd.DataFrame(resultats,columns=colonnes)





#=================================== AXE 2 : PORTEFEUILLE OPTIMAL ===================================================================


def stats_du_portefeuille_optimal (portefeuille_optimal):
    rentabilité_moyenne=portefeuille_optimal.loc['Rentabilité']
    volatilité_moyenne=portefeuille_optimal.loc['Volatilité']

    return rentabilité_moyenne , volatilité_moyenne

def calculer_rendement_portefeuille_historique(rendements,portefeuille_optimal):
    rendement_portefeuille = 0
    for action in actions :
        Rh=rendements[action]*portefeuille_optimal.loc[action]
        rendement_portefeuille+=Rh
    

    return rendement_portefeuille

def beta_et_medaf_portefeuille_optimal(BETA,portefeuille_optimal:pd.Series,Rm,rf):
    beta_p=0
    for action in actions :
        poids_actif=portefeuille_optimal.loc[action]
        b=poids_actif*BETA[action]
        beta_p+=b
    medaf_p=rf+beta_p*(Rm-rf)
    return beta_p , medaf_p

def alpha_jensen_portefeuille_optimal(rentabilité_moyenne_portefeuille,medaf_p):
    a=rentabilité_moyenne_portefeuille-medaf_p
    return a




df =télécharger_et_afficher_les_données(P)
rendements =calculer_les_rendements_mensuels(df)
moyennes,volatilités =calculer_les_moyennes_et_volatilités(rendements)
moyennes_annualisées , volatilités_annualisées=annualiser_stats(moyennes,volatilités)
rf =importer_et_calculer_le_taux_sans_risque(actif_sans_risque)
BETA =calculer_les_betas(rendements)
Medafs , Rm =calculer_les_rentabilités_espérées_par_le_MEDAF(rf,BETA, moyennes_annualisées)
Ratios_sharpe =calculer_les_ratios_de_sharpe(moyennes_annualisées, volatilités_annualisées, rf)
Alphas =calculer_les_Alphas_de_Jensen(Medafs, moyennes_annualisées)
simulations = simulation_portefeuilles(rendements, moyennes_annualisées, rf, 10000)


#==========================================  PORTEFEUILLE OPTIMAL =============================================================
print('=========== PORTEFEUILLE OPTIMAL :========\n')
portefeuille_optimal = simulations.loc[simulations["Sharpe"].idxmax()]
print(portefeuille_optimal)


# Téléchargement du fichier excel contenant le portefeuille optimal
portefeuille_optimal.to_excel('Portefeuille optimal.xlsx')



rentabilté_moyenne_portefeuille ,volatilité_moyenne_portefeuille = stats_du_portefeuille_optimal(portefeuille_optimal)
rendement_portefeuille_historique = calculer_rendement_portefeuille_historique(rendements,portefeuille_optimal)
beta_p , medaf_p = beta_et_medaf_portefeuille_optimal(BETA, portefeuille_optimal, Rm, rf)
alpha_jensen_p = alpha_jensen_portefeuille_optimal(rentabilté_moyenne_portefeuille, medaf_p)



# ==========================================   PARTIE 3 : VISUALISATION GRAPHIQUE DES RESULTATS ==========================================

def visuel_sml(ax):

    betas = [BETA[a] for a in actions]

    rendements_actifs = [moyennes_annualisées[a] for a in actions]
        
    beta_max = max(max(betas), beta_p) + 0.5

    beta_sml = np.linspace(0, beta_max, 200)

    sml = rf + beta_sml * (Rm - rf)

    

    ax.plot(
        beta_sml,
        sml,
        color="black",
        linewidth=2,
        label="SML")

    scatter = ax.scatter(
        betas,
        rendements_actifs,
        c=Alphas,
        cmap="RdYlGn",
        s=250,
        edgecolors="black"
    )

    x = beta_max * 0.75
    y = rf + x * (Rm - rf)

    ax.annotate(
        "Sous-évaluées",
        xy=(x, y + 0.05),
        color="green",
        fontweight="bold",
        ha="center",
        va="center"
    )

    ax.annotate(
        "Surévaluées",
        xy=(x, y - 0.05),
        color="darkred",
        fontweight="bold",
        ha="center",
        va="center",
    )


    for nom, beta, rendement in zip(noms,betas,rendements_actifs):

        ax.annotate(
            nom,
            (beta, rendement),
            xytext=(5,5),
            textcoords="offset points")

    ax.scatter(
        beta_p,
        rentabilté_moyenne_portefeuille,
        marker="*",
        s=500,
        color="red",
        edgecolor="black",
        label="Portefeuille optimal")

    ax.scatter(
        0,
        rf,
        marker="s",
        color="blue",
        s=120,
        label="Rf")

    ax.scatter(
        1,
        Rm,
        marker="D",
        color="purple",
        s=120,
        label="S&P 500 TR")

    ax.set_title("Security Market Line")

    ax.set_xlabel("Bêta")
    ax.set_ylabel("Rentabilité")

    ax.grid(True)

    ax.legend()

    cbar = ax.figure.colorbar(scatter,ax=ax)

    cbar.set_label("Alpha de Jensen")


# ==========================================

def visuel_rendements_cumules(ax):

    couleurs = [
        "#0078D7",
        "#2E8B57",
        "#DAA520",
        "#FF8C00"]

    for action, nom, couleur in zip(
        actions,
        noms,
        couleurs
    ):

        perf = (
            (1 + rendements[action])
            .cumprod()
            * 100
        )

        ax.plot(
            perf.index,
            perf,
            color=couleur,
            linewidth=2,
            label=nom
        )

    perf_portefeuille = (
        (1 + rendement_portefeuille_historique)
        .cumprod()
        * 100
    )

    ax.plot(
        perf_portefeuille.index,
        perf_portefeuille,
        color="red",
        linewidth=3,
        label="Portefeuille optimal")

    perf_marche = ((1 + rendements["^SP500TR"]).cumprod()* 100)

    ax.plot(
        perf_marche.index,
        perf_marche,
        color="black",
        linewidth=3,
        label="S&P 500 TR")

    ax.set_title("Croissance de 100$ investis")

    ax.set_ylabel("Valeur du portefeuille")

    ax.tick_params(axis="x",rotation=45)

    ax.grid(True)

    ax.legend()


# ==========================================

def visuel_monte_carlo(ax):

    
    #Nuage Monte-Carlo
    
    scatter = ax.scatter(
        simulations["Volatilité"],
        simulations["Rentabilité"],
        c=simulations["Sharpe"],
        cmap="viridis",
        alpha=0.55,
        s=30,
        edgecolors="none")

   
    # Portefeuille optimal
  
    volatlité_moyenne = portefeuille_optimal["Volatilité"]
    rentabilité_moyenne = portefeuille_optimal["Rentabilité"]

    ax.scatter(
        volatlité_moyenne,
        rentabilité_moyenne,
        marker="*",
        s=500,
        color="red",
        edgecolor="black",
        linewidth=1.5,
        zorder=10,
        label="Portefeuille optimal")

   
    # S&P 500
    
    volatilité_marché = volatilités_annualisées["^SP500TR"]
    rendement_moyen_marché = moyennes_annualisées["^SP500TR"]

    ax.scatter(
        volatilité_marché,
        rendement_moyen_marché,
        marker="D",
        color="black",
        s=180,
        zorder=9,
        label="S&P 500 TR")

   
    # CML
    
    pente = (rentabilité_moyenne - rf) / volatlité_moyenne

    volatilités_cml = np.linspace(
        simulations["Volatilité"].min() * 0.90,
        simulations["Volatilité"].max() * 1.05,
        200)

    cml = rf + pente * volatilités_cml

    ax.plot(
        volatilités_cml,
        cml,
        color="crimson",
        linestyle="--",
        linewidth=2,
        label="CML")

    

    
    # Zoom et ajustement des axes :
   
    x_min = min(simulations["Volatilité"].min(),volatilité_marché,volatlité_moyenne)

    x_max = max(simulations["Volatilité"].max(),volatilité_marché,volatlité_moyenne)

    y_min = min(simulations["Rentabilité"].min(),rendement_moyen_marché,rentabilité_moyenne)

    y_max = max(simulations["Rentabilité"].max(),rendement_moyen_marché,rentabilité_moyenne)

    marge_x = (x_max - x_min) * 0.08
    marge_y = (y_max - y_min) * 0.08

    ax.set_xlim(x_min - marge_x,x_max + marge_x)

    ax.set_ylim(y_min - marge_y,y_max + marge_y)

   
    # Habillage
    
    ax.set_title( "10 000 portefeuilles simulés",fontsize=14,pad=10)

    ax.set_xlabel("Volatilité",fontsize=11)

    ax.set_ylabel("Rentabilité",fontsize=11)

    ax.grid(True,linestyle="-",linewidth=0.6,alpha=0.35)

    ax.legend(loc="upper left",frameon=True)

    
    # Barre de couleur
    cbar = ax.figure.colorbar(
        scatter,
        ax=ax)

    cbar.set_label("Ratio de Sharpe")



# DASHBOARD

def dashboard():

    fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(24,8),constrained_layout=True)

    visuel_sml(axes[0])

    visuel_rendements_cumules(axes[1])

    visuel_monte_carlo(axes[2])

    fig.suptitle("Dashboard de simulation CAPM",fontsize=18,fontweight="bold")

    plt.show()



# EXECUTION





dashboard()
