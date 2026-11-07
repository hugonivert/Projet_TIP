function res = evaluerReseau(net, donnees, vraiesClasses, nomsClasses)
% EVALUERRESEAU Évalue un réseau entraîné avec les critères du challenge Food-11.
%
%   res = evaluerReseau(net, donnees, vraiesClasses, nomsClasses)
%
%   Entrées :
%     net           : réseau entraîné (renvoyé par trainnet)
%     donnees       : images à évaluer, par exemple l'augmentedImageDatastore
%                     de validation
%     vraiesClasses : vraies classes de ces images (categorical), dans le même
%                     ordre que donnees, par exemple imgDataValidation.Labels
%     nomsClasses   : noms des classes dans l'ordre des sorties du réseau,
%                     c'est-à-dire categories(imgDataTrain.Labels) (cellstr,
%                     string ou categorical acceptés, par exemple les noms
%                     renvoyés par imagePretrainedNetwork)
%
%   Sortie : structure res avec
%     exactitude       : part des images bien classées (accuracy, entre 0 et 1).
%                        À ne pas confondre avec la précision par classe
%                        utilisée dans le calcul du F1.
%     f1ParClasse      : F1 de chaque classe, dans l'ordre de nomsClasses
%     f1Macro          : moyenne des F1 par classe (mesure utilisée par le serveur)
%     nbParametres     : nombre de paramètres appris par le réseau
%                        (le "Total learnables" de analyzeNetwork)
%     frugalite        : score de classement du challenge,
%                        f1Macro - 0.3 / (1 + 1/R), avec R en millions de
%                        paramètres (seule convention qui redonne les exemples
%                        chiffrés du sujet : 3.5 M et F1 = 0.80 donnent 0.57)
%     matriceConfusion : matrice de confusion (lignes = vraies classes,
%                        colonnes = classes prédites, ordre de nomsClasses)
%     predites         : classes prédites pour chaque image (categorical)
%
%   Fonction commune à tous les réseaux du groupe, pour que les scores soient
%   calculés de la même façon et directement comparables.

    poidsFrugalite = 0.3;   % w dans la formule du sujet

    % Noms de classes sous une forme unique (colonne de cellstr), quel que
    % soit le format reçu
    nomsClasses = cellstr(string(nomsClasses(:)));

    % Prédictions
    scores = minibatchpredict(net, donnees);
    predites = scores2label(scores, nomsClasses);
    vraiesClasses = categorical(vraiesClasses, nomsClasses);
    assert(~any(isundefined(vraiesClasses)), ...
        "evaluerReseau : certaines vraies classes ne figurent pas dans nomsClasses.");

    % Matrice de confusion, dans l'ordre des classes (même pour une classe
    % jamais prédite)
    C = confusionmat(vraiesClasses, predites, Order=nomsClasses);

    % Précision et rappel par classe (formules du cours, diapo 27), puis F1
    vraisPositifs   = diag(C);
    precisionClasse = vraisPositifs ./ sum(C, 1)';   % parmi les images prédites dans la classe
    rappelClasse    = vraisPositifs ./ sum(C, 2);    % parmi les images de la classe
    f1Classe = 2 * precisionClasse .* rappelClasse ./ (precisionClasse + rappelClasse);
    f1Classe(isnan(f1Classe)) = 0;   % classe jamais (bien) prédite : F1 = 0

    % Nombre de paramètres appris
    nbParametres = sum(cellfun(@numel, net.Learnables.Value));
    R = nbParametres / 1e6;   % en millions

    res.exactitude       = mean(predites == vraiesClasses);
    res.f1ParClasse      = f1Classe;
    res.f1Macro          = mean(f1Classe);
    res.nbParametres     = nbParametres;
    res.frugalite        = res.f1Macro - poidsFrugalite / (1 + 1 / R);
    res.matriceConfusion = C;
    res.predites         = predites;
end
