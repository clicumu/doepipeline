import posixpath


def write_pipeline(experimental_setup, yaml_dict, iteration):
    """
    Called once for each iteration of the doe optimization.
    """
    sub_process_dict = {}
    num_exp = len(experimental_setup)

    # Misc (paths and parameters)
    vcf_compare_path = yaml_dict['vcf_compare_path']
    gatk_path = yaml_dict['gatk_path']
    ref_path = yaml_dict['ref_path']
    truth_path = yaml_dict['truth_path']
    start_bam = yaml_dict['start_bam']
    inv_name = yaml_dict['inv_name']
    region = yaml_dict['region']

    for run_order in range(1, num_exp+1):
        # Get the experiment number in the order of the running order
        experiment_settings = experimental_setup.loc[experimental_setup['Run Order'] == run_order]
        exp_num = str(experiment_settings.iloc[0]['Exp No'])

        # Variable settings (from the worksheet)
        minMQ = str(int(experiment_settings.iloc[0]['Minimum Mapping Quality Score']))
        minBQ = str(int(experiment_settings.iloc[0]['Minimum Base Quality Score']))
        minCC = str(int(experiment_settings.iloc[0]['Minimum Confidence Threshold for Calling']))
        minQD = str(int(experiment_settings.iloc[0]['Minimum Quality by Depth QD']))
        minRMSMQ = str(int(experiment_settings.iloc[0]['Minimum RMS Mapping Quality']))

        # Output folder for this experiment
        out_base_path = posixpath.join(
            yaml_dict['kaw_workdir'],
            yaml_dict['inv_name'],
            str(iteration),
            exp_num
        )  # dir: kaw_workdir/inv_name/iteration/exp_num


        raw_calls = posixpath.join(out_base_path, 'raw_calls.vcf')
        raw_calls_snps = posixpath.join(out_base_path, 'raw_snp_calls.vcf')
        filter_expression = "QD < %s || FS > 60.0 || %s || MQRankSum < -12.5 || ReadPosRankSum < -8.0" % (minQD, minRMSMQ)
        filtered_snps = posixpath.join(out_base_path, 'filtered_snp.vcf')

        ### Make commands
        sub_process_dict[exp_num] = {}
        # Call:
        # TODO: specifically tell haplotypecaller to call in 43-47MB region, not only restrict it to chr21
        # TODO: specifically tell haplotypecaller to use only 1 core
        sub_process_dict[exp_num][1] = "\
            java -Xmx4G -jar %s \
                -T HaplotypeCaller \
                -R %s \
                -I %s \
                -L %s \
                -nct 1 \
                --genotyping_mode DISCOVERY \
                --min_mapping_quality_score %s \
                --min_base_quality_score %s \
                --standard_min_confidence_threshold_for_calling %s \
                --standard_min_confidence_threshold_for_emitting %s \
                -o %s" \
                % (gatk_path, ref_path, start_bam, region, minMQ, minBQ, minCC, minCC, raw_calls)

        # Extract SNPs:
        sub_process_dict[exp_num][2] = "\
            java -Xmx4G -jar %s \
                -T SelectVariants \
                -R %s \
                -V %s \
                -selectType SNP \
                -o %s" \
                % (gatk_path, ref_path, raw_calls, raw_calls_snps)

        # Apply hard filter:
        sub_process_dict[exp_num][3] = "\
            java -Xmx4G -jar %s \
                -T VariantFiltration \
                -R %s \
                -V %s \
                --filterExpression '%s' \
                --filterName 'GATK_HARD_FILTER' \
                -o %s" \
                % (gatk_path, ref_path, raw_calls_snps, filter_expression, filtered_snps)

        # Compare Call set with Truth set:
        sub_process_dict[exp_num][4] = "\
            python %s \
                --truth_vcf_path %s \
                --called_vcf_path %s \
                --output_dir %s \
                --file_prefix %s" \
                % (vcf_compare_path, truth_path, filtered_snps, out_base_path, inv_name)

    return sub_process_dict
